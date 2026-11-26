import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv(sys.argv[1])
VALID = {"CORRECT", "PARTIEL", "INCORRECT", "REFUS_JUSTIFIE",
         "REFUS_INJUSTIFIE", "REPONSE_A_TORT", "CLARIFICATION"}

df["verdict_humain"] = df["verdict_humain"].astype(str).str.strip().str.upper()
bad = df[~df["verdict_humain"].isin(VALID)]
if len(bad):
    print("Lignes non remplies ou invalides :", bad["id"].tolist())
    sys.exit()

print(f"{len(df)} questions annotées\n")
print("=== Verdicts par catégorie ===")
print(pd.crosstab(df["category"], df["verdict_humain"], margins=True).to_string())

ans = df[df["expected"] == "answer"]
n = len(ans)
print(f"\n=== Exactitude (questions avec réponse, n={n}) ===")
print("CORRECT            :", round(100 * (ans.verdict_humain == "CORRECT").mean(), 1), "%")
print("CORRECT + PARTIEL  :", round(100 * ans.verdict_humain.isin(["CORRECT", "PARTIEL"]).mean(), 1), "%")
print("INCORRECT          :", round(100 * (ans.verdict_humain == "INCORRECT").mean(), 1), "%")
print("REFUS_INJUSTIFIE   :", round(100 * (ans.verdict_humain == "REFUS_INJUSTIFIE").mean(), 1), "%")
print(ans.groupby("category")["verdict_humain"]
      .apply(lambda s: round(100 * (s == "CORRECT").mean(), 1))
      .rename("% CORRECT").to_string())

ab = df[df["expected"] == "refuse"]
print(f"\n=== Questions sans réponse dans le corpus (n={len(ab)}) ===")
print("REFUS_JUSTIFIE :", round(100 * (ab.verdict_humain == "REFUS_JUSTIFIE").mean(), 1), "%")
print("REPONSE_A_TORT :", round(100 * (ab.verdict_humain == "REPONSE_A_TORT").mean(), 1), "%")

am = df[df["expected"] == "clarify"]
print(f"\n=== Questions ambiguës (n={len(am)}) ===")
print("CLARIFICATION :", round(100 * (am.verdict_humain == "CLARIFICATION").mean(), 1), "%")

human_refus = df["verdict_humain"].isin(["REFUS_JUSTIFIE", "REFUS_INJUSTIFIE"])
print("\n=== Détection automatique des refus vs jugement humain ===")
print(pd.crosstab(human_refus.rename("refus humain"), df["refus_auto"].rename("refus auto")).to_string())
print("Accord :", round(100 * (human_refus == df["refus_auto"]).mean(), 1), "%")

print("\n=== Fidélité automatique moyenne par verdict humain ===")
f = pd.to_numeric(df["fidelite_auto_%"], errors="coerce")
print(f.groupby(df["verdict_humain"]).agg(["count", "mean"]).round(1).to_string())

df.to_csv(sys.argv[1].replace(".csv", "_analyse.csv"), index=False, encoding="utf-8-sig")