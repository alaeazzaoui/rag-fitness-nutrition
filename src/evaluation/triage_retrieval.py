
import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

INPUT_FILE = Path("eval/results/review_all_questions.md")
OUTPUT_FILE = Path("eval/results/triage_retrieval.csv")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Seuils indicatifs à ajuster après inspection
HIGH_THRESHOLD = 0.55
LOW_THRESHOLD = 0.35


def extract_questions_and_passages(content):
    questions = []
    blocks = re.split(r"(?=## Q\d+ \|)", content)

    for block in blocks:
        header = re.search(r"## (Q\d+) \| ([^\n]+)", block)
        question = re.search(r"\*\*(.*?)\*\*", block)

        if not header or not question:
            continue

        passages = re.findall(
            r"- P\d+ \| ([^|]+) \| p\.(\d+) \| (.*)",
            block
        )

        questions.append({
            "id": header.group(1),
            "category": header.group(2).strip(),
            "question": question.group(1).strip(),
            "passages": passages
        })

    return questions


def main():
    print("Chargement du modèle...")
    model = SentenceTransformer(MODEL_NAME)

    content = INPUT_FILE.read_text(encoding="utf-8")
    questions = extract_questions_and_passages(content)

    results = []

    for item in questions:
        if not item["passages"]:
            results.append({
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "status": "Non retrouvée",
                "best_similarity": 0,
                "best_source": "",
                "best_page": ""
            })
            continue

        question_embedding = model.encode(
            item["question"],
            convert_to_tensor=True
        )

        passage_texts = [p[2] for p in item["passages"]]
        passage_embeddings = model.encode(
            passage_texts,
            convert_to_tensor=True
        )

        similarities = util.cos_sim(
            question_embedding,
            passage_embeddings
        )[0]

        best_index = int(similarities.argmax())
        best_score = float(similarities[best_index])

        if best_score >= HIGH_THRESHOLD:
            status = "Pertinente"
        elif best_score >= LOW_THRESHOLD:
            status = "À vérifier"
        else:
            status = "Non retrouvée"

        source, page, _ = item["passages"][best_index]

        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "status": status,
            "best_similarity": round(best_score, 3),
            "best_source": source.strip(),
            "best_page": page
        })

        print(
            f"{item['id']} : {status} "
            f"(similarité : {best_score:.3f})"
        )

    import csv

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nRésultat enregistré : {OUTPUT_FILE}")
    print(f"Questions analysées : {len(results)}")


if __name__ == "__main__":
    main()
