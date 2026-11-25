import sys
from pathlib import Path
import json

from src.generation.generator import retrieve_context

STRATEGY = "recursive"
TOP_K = 5


def main(start, end):

    with open("eval/questions.json", encoding="utf-8") as file:
        questions = json.load(file)["questions"]

    lines = []

    for q in questions:

        number = int(q["id"][1:])

        if not (start <= number <= end):
            continue

        lines.append(f"\n## {q['id']} | {q['category']}")
        lines.append(f"**{q['question']}**\n")

        passages = retrieve_context(
            q["question"],
            strategy=STRATEGY,
            top_k=TOP_K
        )

        for i, p in enumerate(passages, 1):

            text = " ".join(p["content"].split())[:350]

            lines.append(
                f"- P{i} | {p['source']} | p.{p['page']} | {text}"
            )

    out = Path(f"eval/results/review_Q{start:02d}_Q{end:02d}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved to: {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))