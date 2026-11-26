import sys
import re
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")

# Métadonnées fournies (à confirmer par le DOI lu dans le PDF)
PROVIDED = {
    "1550-2783-4-8.pdf": ("International Society of Sports Nutrition Position Stand: Protein and Exercise", "Bill Campbell et al.", 2007, "10.1186/1550-2783-4-8"),
    "BJSM-IOC-consensus-statement-on-Relative-Energy-Deficiency-in-Sport-REDs.pdf": ("2023 International Olympic Committee's (IOC) Consensus Statement on Relative Energy Deficiency in Sport (REDs)", "Margo Mountjoy et al.", 2023, "10.1136/bjsports-2023-106994"),
    "bjsports-2023-106807.pdf": ("Resistance Training Prescription for Muscle Strength and Hypertrophy in Healthy Adults: A Systematic Review and Bayesian Network Meta-analysis", "Brad S. Currier et al.", 2023, "10.1136/bjsports-2023-106807"),
    "fspor-04-949021.pdf": ("Resistance Training Variables for Optimization of Muscle Hypertrophy: An Umbrella Review", "R. Bernárdez-Vázquez, J. Raya-González, D. Castillo et al.", 2022, "10.3389/fspor.2022.949021"),
    "s12970-015-0090-y.pdf": ("International Society of Sports Nutrition Position Stand: Beta-Alanine", "Eric T. Trexler et al.", 2015, "10.1186/s12970-015-0090-y"),
    "s12970-017-0174-y.pdf": ("International Society of Sports Nutrition Position Stand: Diets and Body Composition", "Alan A. Aragon et al.", 2017, "10.1186/s12970-017-0174-y"),
    "s12970-017-0177-8.pdf": ("International Society of Sports Nutrition Position Stand: Protein and Exercise", "Ralf Jäger et al.", 2017, "10.1186/s12970-017-0177-8"),
    "s12970-017-0189-4.pdf": ("International Society of Sports Nutrition Position Stand: Nutrient Timing", "Chad M. Kerksick et al.", 2017, "10.1186/s12970-017-0189-4"),
    "s12970-017-0192-9.pdf": ("Vegan Diets: Practical Advice for Athletes and Exercisers", "David Rogerson", 2017, "10.1186/s12970-017-0192-9"),
    "s12970-020-00383-4.pdf": ("International Society of Sports Nutrition Position Stand: Caffeine and Exercise Performance", "Nanci S. Guest et al.", 2021, "10.1186/s12970-020-00383-4"),
    "s40279-025-02285-4.pdf": ("Dietary Recommendations for Body Mass and Composition Manipulation in Male and Female Athletes: A Scoping Review of Consensus Statements, Position Stands and Practice Guidelines from International Expert Groups", "Lauren V. Delany, Nessan Costello, Ben Jones, Susan H. Backhouse", 2025, "10.1007/s40279-025-02285-4"),
}

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>,;]+")
LIC_RE = re.compile(r"(Creative Commons[^\n]{0,140}|CC[ -]BY(?:-NC)?(?:-ND)?(?: ?\d\.\d)?)", re.I)
YEAR_RE = re.compile(r"(?:©|Copyright|Published|Accepted)[^\n]{0,40}?((?:19|20)\d{2})")


def biggest_text(page):
    spans = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                t = s["text"].strip()
                if len(t) >= 3:
                    spans.append((s["size"], t))
    if not spans:
        return ""
    m = max(s for s, _ in spans)
    return " ".join(t for s, t in spans if abs(s - m) < 0.5)[:250]


rows = []
for pdf in sorted(root.rglob("*.pdf")):
    doc = fitz.open(pdf)
    n = len(doc)
    idx = sorted(set(list(range(min(3, n))) + list(range(max(0, n - 2), n))))
    text = "\n".join(doc[i].get_text() for i in idx)
    head = "\n".join(doc[i].get_text() for i in range(min(2, n)))
    text_nows = re.sub(r"\s+", "", text).lower()
    dois = {d.rstrip(".)").lower() for d in DOI_RE.findall(text)}
    first_doi = next((d.rstrip(".)").lower() for d in DOI_RE.findall(head)), "")
    lic = sorted({m.strip() for m in LIC_RE.findall(text)})
    licence = " | ".join(lic)[:200] or "non trouvée (3 premières et 2 dernières pages)"

    if pdf.name in PROVIDED:
        titre, auteurs, annee, doi = PROVIDED[pdf.name]
        ok = doi.lower() in text_nows
        statut = "DOI confirmé dans le PDF" if ok else "À VÉRIFIER (DOI absent ou différent dans le PDF)"
    else:
        titre = (doc.metadata or {}).get("title") or ""
        if len(titre) < 15:
            titre = biggest_text(doc[0])
        auteurs, doi = "", first_doi
        m = YEAR_RE.search(text)
        annee = int(m.group(1)) if m else ""
        statut = "À COMPLÉTER (titre et année extraits automatiquement, auteurs manquants)"

    rows.append({"fichier": pdf.name, "titre": titre, "auteurs": auteurs, "annee": annee,
                 "doi": doi, "url": f"https://doi.org/{doi}" if doi else "",
                 "licence": licence, "statut": statut})

df = pd.DataFrame(rows)
out = Path("data/sources.csv")
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"{len(df)} PDF traités -> {out}\n")
print(df["statut"].value_counts().to_string())
print()
print(df[["fichier", "annee", "doi", "licence", "statut"]].to_string(index=False, max_colwidth=45))