import json
import sys
from datetime import datetime
from pathlib import Path

from src.hallucination import hallucination_detector as hd
from src.generation.generator import retrieve_context


def main(run_file):

    with open(run_file, encoding="utf-8") as file:
        run = json.load(file)

    config = run["config"]

    passages = retrieve_context(
        config["question"],
        strategy=config["strategy"],
        top_k=config["top_k"]
    )

    output = {
        "source_run": str(run_file),
        "config": config,
        "rescored": []
    }

    for use_rule in (True, False):

        hd.USE_NEGATION_RULE = use_rule

        print("\n" + "=" * 70)
        print(f"NEGATION RULE: {'ON' if use_rule else 'OFF'}")
        print("=" * 70)

        for item in run["results"]:

            report = hd.detect_hallucinations(
                item["answer"],
                passages
            )

            print(
                f"\n{item['model']} | "
                f"faithfulness = {report['faithfulness_score']}% | "
                f"S={report['supported']} "
                f"U={report['unsupported']} "
                f"C={report['contradicted']}"
            )

            for number, claim in enumerate(report["claims"], 1):
                print(
                    f"  claim {number}: {claim['status']:<12} "
                    f"ent={claim['scores']['entailment']:<8} "
                    f"contr={claim['scores']['contradiction']}"
                )

            # Vérification de reproductibilité (règle active)
            if use_rule:
                saved = [c["status"] for c in item["claims"]]
                now = [c["status"] for c in report["claims"]]
                print(
                    "  reproduction:",
                    "OK" if saved == now else "DIFFERENT"
                )

            output["rescored"].append({
                "model": item["model"],
                "negation_rule": use_rule,
                "faithfulness_score": report["faithfulness_score"],
                "supported": report["supported"],
                "unsupported": report["unsupported"],
                "contradicted": report["contradicted"],
                "claims": report["claims"]
            })

    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"rescore_{stamp}.json"

    with open(out_file, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=4)

    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python -m src.evaluation.rescore_run <run_file.json>")
    else:
        main(sys.argv[1])