import pandas as pd
import json
from rapidfuzz import fuzz

# =========================
# LOAD DATA
# =========================
with open("nodes_final_fixed.json") as f:
    nodes = json.load(f)

df = pd.DataFrame(nodes)

print("Columns:", df.columns)
print("Total records:", len(df))


# =========================
# MEDICAL DICTIONARY
# =========================
medical_dict = {
    "lung": ("C0024109", ["lung", "pulmonary"]),
    "heart": ("C0018787", ["heart", "cardiac"]),
    "opacity": ("C0032516", ["opacity", "infiltrate"]),
    "effusion": ("C0014867", ["effusion", "fluid"]),
    "pneumonia": ("C0032285", ["pneumonia"]),
    "fracture": ("C0016658", ["fracture", "break"]),
}


# =========================
# CUI MAPPING FUNCTION
# =========================
def map_cui(text):
    text = str(text).lower()

    best_cui = None
    best_score = 0

    for concept, (cui, keywords) in medical_dict.items():
        score = 0

        for word in keywords:
            # Exact match boost
            if word in text:
                score += 2

            # Fuzzy match
            fuzzy_score = fuzz.partial_ratio(word, text)
            if fuzzy_score > 80:
                score += 1

        if score > best_score:
            best_score = score
            best_cui = cui

    return best_cui, best_score


# =========================
# PROCESS ALL TEXT
# =========================
mapped = []
unmapped = []

for text in df["text"]:
    cui, score = map_cui(text)

    if cui:
        mapped.append({
            "text": text,
            "cui": cui,
            "confidence_score": score
        })
    else:
        unmapped.append({
            "text": text
        })


# =========================
# SAVE OUTPUT
# =========================
mapped_df = pd.DataFrame(mapped)
unmapped_df = pd.DataFrame(unmapped)

mapped_df.to_csv("mapped_cui.csv", index=False)
unmapped_df.to_csv("no_cui.csv", index=False)

print("\n✅ Mapping completed!")
print("Mapped:", len(mapped_df))
print("Unmapped:", len(unmapped_df))