import pandas as pd
import json
from radgraph import RadGraph

# =========================
# STEP 1: LOAD CSV
# =========================
csv_path = "C:/radgraph_models/labels.csv"   # 🔥 change if needed

df = pd.read_csv(csv_path)

print("Columns:", df.columns)
print(df.head())

# 🔥 CHANGE THIS COLUMN NAME if needed
TEXT_COLUMN = "report"   # <-- modify based on your CSV

reports = df[TEXT_COLUMN].fillna("").tolist()

# =========================
# STEP 2: INIT RADGRAPH
# =========================
print("\nLoading RadGraph...")
radgraph = RadGraph(model="radgraph", cuda=False)

# =========================
# STEP 3: RUN RADGRAPH
# =========================
def run_radgraph(reports, batch_size=8):
    all_outputs = []

    for i in range(0, len(reports), batch_size):
        batch = reports[i:i+batch_size]

        try:
            output = radgraph(batch)
            all_outputs.append(output)
        except Exception as e:
            print(f"Error at batch {i}: ", e)
            continue

        print(f"{i+len(batch)}/{len(reports)} processed")

    return all_outputs


# 🔥 TEST FIRST (change later to full dataset)
results = run_radgraph(reports, batch_size=16)

# =========================
# STEP 4: SAVE RAW OUTPUT
# =========================
with open("radgraph_output.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved radgraph_output.json")

# =========================
# STEP 5: CONVERT TO GRAPH
# =========================
all_nodes = []
all_edges = []

import json

for item in results:

    # 🔥 If string → convert to dict
    if isinstance(item, str):
        try:
            item = json.loads(item)
        except:
            continue

    # 🔥 If still not dict → skip
    if not isinstance(item, dict):
        continue

    # Now safe
    for idx, report in item.items():

        if not isinstance(report, dict):
            continue

        entities = report.get("entities", {})

        for eid, ent in entities.items():
            all_nodes.append({
                "report_id": idx,
                "entity_id": eid,
                "text": ent.get("tokens", ""),
                "type": ent.get("label", "")
            })

            for rel in ent.get("relations", []):
                all_edges.append({
                    "report_id": idx,
                    "source": eid,
                    "target": rel[1],
                    "relation": rel[0]
                })

# =========================
# STEP 6: SAVE GRAPH
# =========================
nodes_df = pd.DataFrame(all_nodes)
edges_df = pd.DataFrame(all_edges)

nodes_df.to_csv("nodes.csv", index=False)
edges_df.to_csv("edges.csv", index=False)

print("\nSaved nodes.csv and edges.csv")

# =========================
# DONE
# =========================print("\nDEBUG OUTPUT:")
print(type(results))
print(results[:2])   # show first 2 items
print("\n✅ Pipeline completed successfully!")
