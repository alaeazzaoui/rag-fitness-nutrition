import re
import sys
import json
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/results/notebook_results.json")
records = json.load(open(path, encoding="utf-8"))

CITE = re.compile(r"\[S\d+|\bdocuments?\s+S\d+\b", re.I)
SOFT = re.compile(
    r"il n'y a pas de (?:mention|information|précision|donnée)"
    r"|ne (?:mentionnent|contiennent|fournissent|précisent|donnent) pas"
    r"|pas d'informations? (?:suffisantes?|spécifiques?)",
    re.I,
)
MARK = "ne dispose pas d'informations suffisantes"


def norm(text):
    return text.replace("’", "'")


d = pd.DataFrame(records)
d["cite_v1"] = d["has_citation"]
d["cite_v2"] = d["answer"].apply(lambda a: bool(CITE.search(norm(a))))
d["refus_exact"] = d["answer"].apply(lambda a: MARK in norm(a).lower())
d["refus_soft"] = d["answer"].apply(lambda a: bool(SOFT.search(norm(a))))
d["refus_large"] = d["refus_exact"] | d["refus_soft"]

cfg = ["model", "strategy"]

print(f"Fichier : {path} | {len(d)} enregistrements\n")

print("=== Citations (réponses non refusées, %) ===")
cit = (d[~d.refus_exact].groupby(cfg)[["cite_v1", "cite_v2"]].mean() * 100).round(1)
cit.columns = ["[S#] seul (ancienne)", "[S#] ou 'document S#' (corrigée)"]
print(cit.to_string())


def refus_table(sub):
    t = sub.groupby(cfg).agg(
        n=("id", "count"),
        exact=("refus_exact", "sum"),
        large=("refus_large", "sum"),
    )
    t["% exact"] = (100 * t.exact / t.n).round(1)
    t["% élargi"] = (100 * t.large / t.n).round(1)
    return t


print("\n=== Questions SANS réponse dans le corpus (refus attendu) ===")
print(refus_table(d[d.expected == "refuse"]).to_string())

print("\n=== Questions AVEC réponse attendue : refus (à surveiller) ===")
print(refus_table(d[d.expected == "answer"]).to_string())

print("\n=== Refus 'souples' sur questions sans réponse (à relire) ===")
chk = d[(d.expected == "refuse") & d.refus_soft & ~d.refus_exact]
for _, r in chk.iterrows():
    print(f"[{r['id']}] {r['model']} / {r['strategy']} : {r['answer'][:140]}".replace("\n", " "))

out = path.with_name("notebook_results_v2.json")
d.to_json(out, orient="records", force_ascii=False, indent=1)
print("\nSauvegardé :", out)