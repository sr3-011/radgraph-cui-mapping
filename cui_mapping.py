import pandas as pd
import json
import sys
from rapidfuzz import fuzz

# =========================
# LOAD DATA (Flexible input)
# =========================
file_path = sys.argv[1] if len(sys.argv) > 1 else "nodes_final_fixed.json"

with open(file_path) as f:
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

    for _, (cui, keywords) in medical_dict.items():
        score = 0

        for word in keywords:
            # Exact match boost
            if word in text:
                score += 2

            # Fuzzy match
            if fuzz.partial_ratio(word, text) > 80:
                score += 1

        if score > best_score:
            best_score = score
            best_cui = cui

    return best_cui, best_score


# =========================
# PROCESS DATA
# =========================
def process_data(df):
    mapped = []
    unmapped = []

    for i, row in df.iterrows():
        text = row.get("text", "")
        cui, score = map_cui(text)

        base_data = {
            "report_id": row.get("report_id"),
            "entity_id": row.get("entity_id"),
            "text": text,
            "type": row.get("type")
        }

        # Optional fields
        if "subject_id" in df.columns:
            base_data["subject_id"] = row.get("subject_id")

        if "study_id" in df.columns:
            base_data["study_id"] = row.get("study_id")

        if cui:
            base_data["cui"] = cui
            base_data["confidence_score"] = score
            mapped.append(base_data)
        else:
            unmapped.append(base_data)

        # Progress print (every 8000 rows)
        if i % 8000 == 0:
            print(f"Processed {i} rows...")

    return mapped, unmapped


mapped, unmapped = process_data(df)


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