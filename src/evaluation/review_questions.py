import json
from pathlib import Path

from src.generation.generator import retrieve_context

STRATEGY = "recursive"
TOP_K = 15
TOTAL_QUESTIONS = 60


def main():

    with open("eval/questions.json", encoding="utf-8") as file:
        questions = json.load(file)["questions"]

    lines = []

    for index, q in enumerate(questions, 1):

        print(f"Processing {q['id']} ({index}/{len(questions)})...")

        lines.append(f"\n## {q['id']} | {q['category']}")
        lines.append(f"**{q['question']}**\n")

        try:
            passages = retrieve_context(
                q["question"],
                strategy=STRATEGY,
                top_k=TOP_K
            )

            for i, p in enumerate(passages, 1):
                text = " ".join(p["content"].split())[:1500]

                lines.append(
                    f"- P{i} | {p['source']} | p.{p['page']} | {text}"
                )

        except Exception as e:
            lines.append(f"ERROR: {str(e)}")

    out = Path("eval/results/review_all_questions.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nAnalysis completed: {out}")


if __name__ == "__main__":
    main()