import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENTAILMENT_THRESHOLD = 0.85
CONTRADICTION_THRESHOLD = 0.80
MIN_MARGIN = 0.15

USE_NEGATION_RULE = False


# ============================================================
# LOAD NLI MODEL
# ============================================================

print("Loading NLI model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
).to(DEVICE)

model.eval()

LABELS = {
    label.lower(): index
    for index, label in model.config.id2label.items()
}

ENTAILMENT_ID = LABELS["entailment"]
CONTRADICTION_ID = LABELS["contradiction"]

print(f"NLI model loaded successfully on {DEVICE}")


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def split_into_claims(answer):

    if not answer:
        return []

    text = re.sub(r"\[S\d+\]", "", answer)

    # Preserve line boundaries to separate numbered claims
    lines = text.splitlines()

    claims = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Remove Markdown formatting
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"\*(.*?)\*", r"\1", line)

        # Remove list numbering and bullets
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)

        # Split sentences
        sentences = re.split(r"(?<=[.!?])\s+", line)

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence.split()) < 5:
                continue

            if sentence[-1] not in ".!?":
                continue

            claims.append(sentence)

    return claims


# ============================================================
# NEGATION DETECTION
# ============================================================

NEGATION_PATTERNS = [
    r"\bne\b.*\bpas\b",
    r"\bn['’]est pas\b",
    r"\bpas systématiquement\b",
    r"\bpas nécessairement\b",
    r"\bni\b",
    r"\baucun(?:e)?\b",
    r"\babsence de\b",
    r"\bsans\b",
    r"\bnon observé\b",
    r"\bnon démontré\b"
]


def has_negation_conflict(claim, passages):

    """
    Detects whether a passage contains a negation
    in a sentence discussing the same subject.

    A detected negation prevents automatic validation.
    It does not automatically mean contradiction.
    """

    claim_words = {
        word.lower()
        for word in re.findall(r"\b[a-zA-ZÀ-ÿ]{5,}\b", claim)
    }

    for passage in passages:

        content = passage.get("content", "")

        sentences = re.split(r"(?<=[.!?])\s+", content)

        for sentence in sentences:

            sentence_words = {
                word.lower()
                for word in re.findall(
                    r"\b[a-zA-ZÀ-ÿ]{5,}\b",
                    sentence
                )
            }

            common_words = claim_words.intersection(sentence_words)

            # Require at least one meaningful shared term
            if not common_words:
                continue

            has_negation = any(
                re.search(pattern, sentence.lower())
                for pattern in NEGATION_PATTERNS
            )

            if has_negation:
                return True

    return False


# ============================================================
# NLI SCORING
# ============================================================

def calculate_nli_scores(claim, passage):

    # Normalisation des espaces et retours à la ligne issus des PDF
    premise = " ".join(passage.get("content", "").split())
    claim = " ".join(claim.split())

    inputs = tokenizer(
        premise,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    scores = {
        label.lower(): float(probabilities[index])
        for index, label in model.config.id2label.items()
    }

    return scores


# ============================================================
# CHECK ONE CLAIM
# ============================================================

def check_claim(claim, passages):

    best_entailment = 0.0
    best_contradiction = 0.0

    best_evidence = None

    for passage in passages:

        scores = calculate_nli_scores(claim, passage)

        entailment = scores["entailment"]
        contradiction = scores["contradiction"]

        if entailment > best_entailment:

            best_entailment = entailment
            best_evidence = passage

        best_contradiction = max(
            best_contradiction,
            contradiction
        )

    negation_detected = (
        has_negation_conflict(claim, passages)
        if USE_NEGATION_RULE
        else False
    )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if best_contradiction >= CONTRADICTION_THRESHOLD:

        status = "CONTRADICTED"

        confidence = best_contradiction

        evidence = best_evidence

    elif (
        best_entailment >= ENTAILMENT_THRESHOLD
        and best_entailment - best_contradiction >= MIN_MARGIN
        and not negation_detected
    ):

        status = "SUPPORTED"

        confidence = best_entailment

        evidence = best_evidence

    else:

        status = "UNSUPPORTED"

        confidence = best_entailment

        evidence = None

    return {
        "text": claim,
        "status": status,
        "confidence": round(confidence, 4),
        "evidence": {
            "source": best_evidence.get("source"),
            "page": best_evidence.get("page")
        } if evidence else None,
        "scores": {
            "entailment": round(best_entailment, 4),
            "contradiction": round(best_contradiction, 4)
        },
        "negation_detected": negation_detected
    }


# ============================================================
# DETECT HALLUCINATIONS
# ============================================================

def detect_hallucinations(answer, passages):

    claims = split_into_claims(answer)

    results = []

    for claim in claims:

        result = check_claim(
            claim,
            passages
        )

        results.append(result)

    total = len(results)

    supported = sum(
        1 for r in results
        if r["status"] == "SUPPORTED"
    )

    unsupported = sum(
        1 for r in results
        if r["status"] == "UNSUPPORTED"
    )

    contradicted = sum(
        1 for r in results
        if r["status"] == "CONTRADICTED"
    )

    faithfulness = (
        supported / total * 100
        if total > 0
        else 0
    )

    return {
        "total_claims": total,
        "supported": supported,
        "unsupported": unsupported,
        "contradicted": contradicted,
        "faithfulness_score": round(faithfulness, 2),
        "claims": results
    }


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(results):

    print("\n========== HALLUCINATION DETECTION ==========")

    print("Total claims:", results["total_claims"])
    print("Supported:", results["supported"])
    print("Unsupported:", results["unsupported"])
    print("Contradicted:", results["contradicted"])

    print(
        "\nFaithfulness score:",
        results["faithfulness_score"],
        "%"
    )

    for index, claim in enumerate(results["claims"], 1):

        print(f"\n--- CLAIM {index} ---")

        print("Text:", claim["text"])
        print("Status:", claim["status"])
        print("Confidence:", claim["confidence"])

        if claim["evidence"]:

            print(
                "Evidence:",
                claim["evidence"]["source"],
                "| Page:",
                claim["evidence"]["page"]
            )

        print("Scores:", claim["scores"])

        print(
            "Negation detected:",
            claim["negation_detected"]
        )