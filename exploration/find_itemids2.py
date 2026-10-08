import os, sys
import pandas as pd

base = os.path.expanduser(sys.argv[1])
items = pd.read_csv(os.path.join(base, "icu", "d_items.csv.gz"))
labs = pd.read_csv(os.path.join(base, "hosp", "d_labitems.csv.gz"))

item_terms = {
    "vent_procedures": r"ventilation|intubat|extubat|endotracheal|\bett\b|high flow|hfnc|nasal cannula|mechanical vent",
    "map_other": r"art bp|\babp\b|\bnbp\b|bp mean|\bmap\b",
    "sedation": r"\brass\b|richmond|sedation|propofol|midazolam|dexmedetomidine|precedex|lorazepam|ativan|ketamine|fentanyl|versed|diprivan",
    "delirium_drugs_icu": r"quetiapine|olanzapine|risperidone|ziprasidone|seroquel|zyprexa",
}
urine_pat = r"urine|irrigant|suprapubic|ureteral|ileoconduit|conduit|cystostomy|nephrostomy|void|foley|incontinen"

rows = []
for concept, pat in item_terms.items():
    m = items[items["label"].str.contains(pat, case=False, regex=True, na=False)]
    for _, r in m.iterrows():
        rows.append(["d_items", concept, r["itemid"], r["label"], r["linksto"], r["category"], r.get("unitname", "")])

m = items[(items["linksto"] == "outputevents") &
          items["label"].str.contains(urine_pat, case=False, regex=True, na=False)]
for _, r in m.iterrows():
    rows.append(["d_items", "urine_output_all", r["itemid"], r["label"], r["linksto"], r["category"], r.get("unitname", "")])

m = labs[labs["label"].str.contains(r"specimen|delivery|device|fio2|oxygen|lactate", case=False, regex=True, na=False)]
for _, r in m.iterrows():
    rows.append(["d_labitems", "lab_context", r["itemid"], r["label"], r["fluid"], r["category"], ""])

out = pd.DataFrame(rows, columns=["dictionary", "concept", "itemid", "label", "linksto_or_fluid", "category", "unit"])
out = out.drop_duplicates(subset=["dictionary", "concept", "itemid"])
out.to_csv("sofa_candidates2.csv", index=False)
print(out.to_string(index=False))
print("\nSaved sofa_candidates2.csv with", len(out), "rows")
