import gc
import time
import torch
import chromadb

from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.hallucination.hallucination_detector import (
    detect_hallucinations,
    display_results
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

DB_DIR = Path("data/chroma_db")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

GENERATION_MODELS = {
    "Qwen 0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen 1.5B": "Qwen/Qwen2.5-1.5B-Instruct"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K = 5
MAX_NEW_TOKENS = 200

STRATEGY = "recursive"

QUESTION = (
    "Quels risques une consommation excessive de protéines "
    "peut-elle présenter pour les reins et le foie ?"
)


# ==================================================
# 2. INITIALISATION
# ==================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
    device=DEVICE
)

client = chromadb.PersistentClient(
    path=str(DB_DIR)
)

print("Embedding model loaded!")


# ==================================================
# 3. RETRIEVAL
# ==================================================

def retrieve_context(
    question,
    strategy="recursive",
    top_k=5
):

    collection_name = f"fitness_{strategy}"

    collection = client.get_collection(
        name=collection_name
    )

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    passages = []

    for i, content in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i]

        passages.append({

            "content": content,

            "source": metadata.get(
                "source",
                "Unknown"
            ),

            "page": metadata.get(
                "page",
                "Unknown"
            ),

            "distance": results["distances"][0][i]

        })

    return passages


# ==================================================
# 4. AFFICHAGE DU CONTEXTE
# ==================================================

def display_context(passages):

    print("\n========== RETRIEVED CONTEXT ==========")

    for i, passage in enumerate(
        passages,
        start=1
    ):

        print(f"\n--- PASSAGE {i} ---")

        print("Source:", passage["source"])
        print("Page:", passage["page"])

        print(
            "Distance:",
            round(passage["distance"], 4)
        )

        print(
            passage["content"][:800]
        )


# ==================================================
# 5. CONSTRUCTION DU PROMPT
# ==================================================

def build_prompt(question, passages):

    context = ""

    for i, passage in enumerate(
        passages,
        start=1
    ):

        context += (
            f"\n[S{i}] "
            f"Source: {passage['source']}, "
            f"Page: {passage['page']}\n"
            f"{passage['content']}\n"
        )

    prompt = f"""
Tu es un assistant spécialisé en nutrition et activité physique.

Ta mission est de répondre à la question en utilisant
exclusivement les documents fournis.

RÈGLES OBLIGATOIRES :

1. Utilise uniquement les informations des documents.
2. N'invente aucune information.
3. Ne complète pas les informations avec tes connaissances personnelles.
4. Cite les sources avec [S1], [S2], etc.
5. Si les documents ne permettent pas de répondre,
   indique :
   "Je ne dispose pas d'informations suffisantes
   dans les documents pour répondre à cette question."
6. Ne formule pas de recommandations médicales personnalisées.
7. Réponds en français.
8. Sois précis et concis.
9. N'ajoute aucun chiffre ou effet absent des documents.
10. Ne présente pas une hypothèse comme un fait établi.

DOCUMENTS :
{context}

QUESTION :
{question}

RÉPONSE :
"""

    return prompt


# ==================================================
# 6. CHARGEMENT DU MODÈLE
# ==================================================

def load_generation_model(model_name):

    print(
        f"\nLoading generation model: {model_name}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32
    )

    model.to(DEVICE)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Generation model loaded!")

    return tokenizer, model


# ==================================================
# 7. GÉNÉRATION DE LA RÉPONSE
# ==================================================

def generate_answer(
    question,
    passages,
    tokenizer,
    model
):

    prompt = build_prompt(
        question,
        passages
    )

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    # Formatage du prompt
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenisation explicite
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    start_time = time.time()

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    generation_time = time.time() - start_time

    # Extraction des nouveaux tokens uniquement
    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[0][input_length:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return answer, generation_time


# ==================================================
# 8. ÉVALUATION NLI
# ==================================================

def evaluate_answer(answer, passages):

    report = detect_hallucinations(
        answer,
        passages
    )

    display_results(report)

    return report


# ==================================================
# 9. COMPARAISON DES MODÈLES
# ==================================================

def compare_models(question, passages):

    comparison_results = []

    for model_label, model_name in GENERATION_MODELS.items():

        print("\n" + "=" * 60)

        print(
            f"TESTING MODEL: {model_label}"
        )

        print("=" * 60)

        tokenizer = None
        model = None

        try:

            # Charger le modèle
            tokenizer, model = load_generation_model(
                model_name
            )

            # Générer la réponse
            answer, generation_time = generate_answer(
                question,
                passages,
                tokenizer,
                model
            )

            print("\n========== GENERATED ANSWER ==========")

            print(answer)

            print(
                "\nGeneration time:",
                round(generation_time, 2),
                "seconds"
            )

            # Détection des hallucinations
            print(
                "\n========== HALLUCINATION EVALUATION =========="
            )

            report = evaluate_answer(
                answer,
                passages
            )

            comparison_results.append({

                "model": model_label,

                "answer": answer,

                "generation_time": round(
                    generation_time,
                    2
                ),

                "faithfulness_score": report[
                    "faithfulness_score"
                ],

                "supported": report["supported"],

                "unsupported": report["unsupported"],

                "contradicted": report["contradicted"],

                "total_claims": report["total_claims"],

                "claims": report["claims"]

            })

        except Exception as error:

            print(
                f"\nERROR with {model_label}: {error}"
            )

        finally:

            # Libérer la mémoire
            if model is not None:
                del model

            if tokenizer is not None:
                del tokenizer

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return comparison_results


# ==================================================
# 10. TABLEAU COMPARATIF
# ==================================================

def display_comparison(results):

    print("\n")
    print("=" * 85)

    print("FINAL MODEL COMPARISON")

    print("=" * 85)

    print(
        f"{'Model':<15}"
        f"{'Time (s)':<15}"
        f"{'Faithfulness':<18}"
        f"{'Supported':<13}"
        f"{'Unsupported':<15}"
        f"{'Contradicted':<15}"
    )

    print("-" * 85)

    for result in results:

        print(
            f"{result['model']:<15}"
            f"{result['generation_time']:<15}"
            f"{str(result['faithfulness_score']) + '%':<18}"
            f"{result['supported']:<13}"
            f"{result['unsupported']:<15}"
            f"{result['contradicted']:<15}"
        )

    print("=" * 85)


# ==================================================
# 11. SAUVEGARDE DES RÉSULTATS
# ==================================================

def save_results(results):

    import json
    from datetime import datetime
    from src.hallucination import hallucination_detector as hd

    output_dir = Path("eval/results")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = output_dir / f"run_{timestamp}.json"

    payload = {
        "config": {
            "question": QUESTION,
            "strategy": STRATEGY,
            "top_k": TOP_K,
            "max_new_tokens": MAX_NEW_TOKENS,
            "entailment_threshold": hd.ENTAILMENT_THRESHOLD,
            "contradiction_threshold": hd.CONTRADICTION_THRESHOLD,
            "min_margin": hd.MIN_MARGIN
        },
        "results": results
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(
        f"\nResults saved to: {output_file}"
    )


# ==================================================
# 12. MAIN
# ==================================================

if __name__ == "__main__":

    print("\n========== RAG FITNESS & NUTRITION ==========")

    print("\nDevice:", DEVICE)

    print("\nQuestion:", QUESTION)

    # Récupération unique pour les deux modèles
    passages = retrieve_context(
        QUESTION,
        strategy=STRATEGY,
        top_k=TOP_K
    )

    if not passages:

        print("\nNo relevant documents found.")

    else:

        display_context(passages)

        # Comparaison des deux modèles
        results = compare_models(
            QUESTION,
            passages
        )

        if results:

            display_comparison(results)

            save_results(results)

        else:

            print(
                "\nNo model produced a valid result."
            )