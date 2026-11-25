import json
from collections import Counter

with open("eval/questions.json", encoding="utf-8") as f:
    qs = json.load(f)["questions"]

print("Total :", len(qs))
print("Statuts :", dict(Counter(q["validation_status"] for q in qs)))
print("Comportements :", dict(Counter(q["expected_behavior"] for q in qs)))

to_fill = [q for q in qs if q["expected_behavior"] == "answer"]

no_ref = [q["id"] for q in to_fill if not q["reference_answer"].strip()]
no_src = [q["id"] for q in to_fill if not q["gold_sources"]]

print(f"\nQuestions 'answer' : {len(to_fill)}")
print("Sans réponse de référence :", no_ref)
print("Sans source/page :", no_src)