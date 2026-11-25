import pymupdf
import json
from pathlib import Path

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

documents = []

for pdf_path in RAW_DIR.glob("*.pdf"):
    doc = pymupdf.open(pdf_path)

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()

        if text:
            documents.append({
                "source": pdf_path.name,
                "page": page_number,
                "content": text
            })

    print(f"✅ {pdf_path.name} | {len(doc)} pages")

output_file = OUTPUT_DIR / "documents.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=4)

print(f"\nExtraction terminée : {len(documents)} pages extraites.")
print(f"Résultat : {output_file}")
