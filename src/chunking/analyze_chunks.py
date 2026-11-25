import json
import pandas as pd
from pathlib import Path

CHUNKS_DIR = Path("data/chunks")
RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

strategies = ["fixed", "recursive", "semantic"]

results = []

for strategy in strategies:

    file_path = CHUNKS_DIR / f"{strategy}_chunks.json"

    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    lengths = [len(chunk["content"]) for chunk in chunks]

    results.append({
        "Strategy": strategy,
        "Total Chunks": len(chunks),
        "Average Length": round(sum(lengths) / len(lengths), 2),
        "Min Length": min(lengths),
        "Max Length": max(lengths),
        "Total Characters": sum(lengths),
        "Unique Documents": len(set(c["source"] for c in chunks))
    })

df = pd.DataFrame(results)

print("\n===== CHUNKING COMPARISON =====\n")
print(df.to_string(index=False))

output_file = RESULTS_DIR / "chunking_comparison.csv"
df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")
