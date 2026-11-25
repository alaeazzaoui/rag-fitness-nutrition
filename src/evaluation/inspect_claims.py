import json
import re
import sys

from src.hallucination import hallucination_detector as hd
from src.generation.generator import retrieve_context


WINDOW_SIZES = (1, 2, 3)


def split_sentences(text):

    text = " ".join(text.split())

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [s.strip() for s in sentences if len(s.split()) >= 4]


def main(run_file, model_label):

    with open(run_file, encoding="utf-8") as file:
        run = json.load(file)

    config = run["config"]

    passages = retrieve_context(
        config["question"],
        strategy=config["strategy"],
        top_k=config["top_k"]
    )

    item = next(
        r for r in run["results"]
        if r["model"] == model_label
    )

    claims = hd.split_into_claims(item["answer"])

    for number, claim in enumerate(claims, 1):

        print("\n" + "=" * 70)
        print(f"CLAIM {number}: {claim}")
        print("=" * 70)

        # Référence : passage entier
        best_whole = None

        for index, passage in enumerate(passages, 1):

            scores = hd.calculate_nli_scores(claim, passage)

            if best_whole is None or scores["entailment"] > best_whole[0]:
                best_whole = (
                    scores["entailment"],
                    scores["contradiction"],
                    index
                )

        print(
            f"\n  Passage entier : ent={best_whole[0]:.4f} "
            f"contr={best_whole[1]:.4f} (P{best_whole[2]})"
        )

        # Fenêtres de phrases consécutives
        for size in WINDOW_SIZES:

            best = None

            for index, passage in enumerate(passages, 1):

                sentences = split_sentences(passage["content"])

                if len(sentences) < size:
                    continue

                for start in range(len(sentences) - size + 1):

                    window = " ".join(sentences[start:start + size])

                    scores = hd.calculate_nli_scores(
                        claim,
                        {"content": window}
                    )

                    if best is None or scores["entailment"] > best[0]:
                        best = (
                            scores["entailment"],
                            scores["contradiction"],
                            index,
                            window
                        )

            if best is None:
                print(f"\n  Fenêtre {size} phrase(s) : aucune")
                continue

            print(
                f"\n  Fenêtre {size} phrase(s) : ent={best[0]:.4f} "
                f"contr={best[1]:.4f} (P{best[2]})"
            )
            print(f"    -> {best[3][:220]}")


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print(
            "Usage: python -m src.evaluation.inspect_claims "
            "<run_file.json> <model_label>"
        )
    else:
        main(sys.argv[1], sys.argv[2])