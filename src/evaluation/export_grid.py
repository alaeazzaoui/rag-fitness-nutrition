import sys
import json
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

path = Path(sys.argv[1])
model = sys.argv[2] if len(sys.argv) > 2 else "Qwen 1.5B"
strategy = sys.argv[3] if len(sys.argv) > 3 else "recursive"

records = json.load(open(path, encoding="utf-8"))

rows = []
for r in records:
    if r["model"] != model or r["strategy"] != strategy:
        continue
    claims = r.get("claims") or []
    faith = (round(100 * sum(c["status"] == "SUPPORTED" for c in claims) / len(claims), 1)
             if claims else "")
    rows.append({
        "id": r["id"],
        "category": r["category"],
        "expected": r["expected"],
        "question": r["question"],
        "answer": r["answer"],
        "sources_top5": " | ".join(f"{s[0]} p.{s[1]}" for s in r["top_sources"]),
        "refus_auto": bool(r["refus_large"]),
        "citation_auto": bool(r["cite_v2"]),
        "tronquee": bool(r["truncated"]),
        "fidelite_auto_%": faith,
        "verdict_humain": "",
        "commentaire": "",
    })

df = pd.DataFrame(rows).sort_values("id")
out = path.with_name(f"grille_humaine_{model.replace(' ', '_')}_{strategy}.csv")
df.to_csv(out, index=False, encoding="utf-8-sig")

print(f"{len(df)} questions exportées vers {out}")
print("""
Vocabulaire de verdict_humain :
  CORRECT          réponse juste et appuyée par les documents
  PARTIEL          en partie juste ou incomplète
  INCORRECT        fausse ou non soutenue par les documents
  REFUS_JUSTIFIE   refus alors que le corpus ne contient pas la réponse
  REFUS_INJUSTIFIE refus alors que la réponse existe dans le corpus
  REPONSE_A_TORT   répond alors que le corpus ne contient pas la réponse
  CLARIFICATION    (questions ambiguës) demande une précision ou répond avec réserves
""")