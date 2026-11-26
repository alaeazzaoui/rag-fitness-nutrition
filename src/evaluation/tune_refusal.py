import sys
import argparse

import numpy as np
import pandas as pd
import chromadb

sys.stdout.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("grid", help="CSV de la grille humaine")
ap.add_argument("--db", default="data/chroma_db")
ap.add_argument("--collection", default=None)
ap.add_argument("--model", default="all-MiniLM-L6-v2")
ap.add_argument("--k", type=int, default=5)
args = ap.parse_args()

# 1. Choix de la collection
client = chromadb.PersistentClient(path=args.db)
names = [c if isinstance(c, str) else c.name for c in client.list_collections()]
print("Collections disponibles :")
for n in names:
    print(f"  - {n} ({client.get_collection(n).count()} chunks)")

name = args.collection
if name is None:
    cands = [n for n in names
             if "recursive" in n.lower()
             and not any(w in n.lower() for w in ("multi", "para", "ablation"))]
    if len(cands) != 1:
        print("\nPrécise la collection : --collection NOM")
        sys.exit(0)
    name = cands[0]
print(f"\nCollection utilisée : {name}")
col = client.get_collection(name)

# 2. Distance top-1 de chaque question (sans génération)
df = pd.read_csv(args.grid, encoding="utf-8-sig")[["id", "category", "expected", "question"]]
questions = list(df["question"])
try:
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(args.model).encode(questions, normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=emb, n_results=args.k, include=["distances"])
except ImportError:
    res = col.query(query_texts=questions, n_results=args.k, include=["distances"])

dist = np.array(res["distances"])
df["top1"] = dist.min(axis=1)
df["mean_topk"] = dist.mean(axis=1)

print("\n=== Distance top-1 par catégorie ===")
print(df.groupby("category")["top1"].agg(["count", "mean", "median", "min", "max"]).round(3))

# 3. Balayage du seuil : refus si top1 > seuil
absent = df[df.expected == "refuse"]["top1"].values
answer = df[df.expected == "answer"]["top1"].values
ambig = df[df.expected == "clarify"]["top1"].values

auc = np.mean([(a > b) + 0.5 * (a == b) for a in absent for b in answer])
print(f"\nAUC (distance top-1 sépare sans réponse / avec réponse) : {auc:.3f}")

rows = []
for t in np.arange(np.floor(df.top1.min() * 100) / 100, np.ceil(df.top1.max() * 100) / 100 + 0.01, 0.01):
    ra, ri, rc = (absent > t).mean(), (answer > t).mean(), (ambig > t).mean()
    rows.append({"seuil": round(t, 2),
                 "refus_justifies_%": round(100 * ra, 1),
                 "refus_injustifies_%": round(100 * ri, 1),
                 "refus_ambigues_%": round(100 * rc, 1),
                 "J": round(ra - ri, 3)})
sweep = pd.DataFrame(rows)
print("\n=== Balayage du seuil (refus si distance top-1 > seuil) ===")
print(sweep.to_string(index=False))
best = sweep.loc[sweep.J.idxmax()]
print(f"\nMeilleur compromis (J = refus justifiés - refus injustifiés) : seuil {best.seuil}, "
      f"{best['refus_justifies_%']} % justifiés, {best['refus_injustifies_%']} % injustifiés")

df.to_csv(f"retrieval_distances_{name}.csv", index=False, encoding="utf-8-sig")
sweep.to_csv(f"sweep_refus_{name}.csv", index=False, encoding="utf-8-sig")