import os, sys
import pandas as pd

base = os.path.expanduser(sys.argv[1])
items = pd.read_csv(os.path.join(base, "icu", "d_items.csv.gz"))
labs = pd.read_csv(os.path.join(base, "hosp", "d_labitems.csv.gz"))

item_terms = {
    "gcs": r"gcs",
    "fio2_oxygen": r"inspired o2|fio2",
    "spo2": r"spo2|pulseoxymetry",
    "map_bp": r"arterial blood pressure mean|non invasive blood pressure mean|abp mean|nbp mean",
    "resp_support": r"ventilator|vent mode|o2 delivery|high flow|hfnc|cpap|bipap|niv|endotracheal|tracheostomy",
    "vasoactive": r"norepinephrine|epinephrine|dopamine|dobutamine|vasopressin|phenylephrine|milrinone|angiotensin",
    "mech_support_ecmo": r"ecmo|iabp|intra-aortic|impella|lvad|balloon pump|ventricular assist",
    "rrt": r"dialysis|crrt|cvvh|hemofiltration|renal replacement",
    "weight": r"weight",
    "delirium_related": r"delirium|cam-icu|haloperidol|quetiapine|olanzapine",
}
lab_terms = {
    "bilirubin": r"bilirubin",
    "creatinine": r"creatinine",
    "platelets": r"platelet",
    "potassium": r"potassium",
    "ph": r"^ph$",
    "bicarbonate": r"bicarbonate",
    "po2_pao2": r"^po2$|pao2",
    "oxygen_other": r"oxygen|fio2",
}

rows = []
for concept, pat in item_terms.items():
    m = items[items["label"].str.contains(pat, case=False, regex=True, na=False)
              | items["abbreviation"].str.contains(pat, case=False, regex=True, na=False)]
    for _, r in m.iterrows():
        rows.append(["d_items", concept, r["itemid"], r["label"], r["linksto"], r["category"], r.get("unitname", "")])

uo = items[(items["linksto"] == "outputevents") &
           items["label"].str.contains(r"urine|foley|void|nephrostomy|condom", case=False, regex=True, na=False)]
for _, r in uo.iterrows():
    rows.append(["d_items", "urine_output", r["itemid"], r["label"], r["linksto"], r["category"], r.get("unitname", "")])

for concept, pat in lab_terms.items():
    m = labs[labs["label"].str.contains(pat, case=False, regex=True, na=False)]
    for _, r in m.iterrows():
        rows.append(["d_labitems", concept, r["itemid"], r["label"], r["fluid"], r["category"], ""])

out = pd.DataFrame(rows, columns=["dictionary", "concept", "itemid", "label", "linksto_or_fluid", "category", "unit"])
out.to_csv("sofa_candidates.csv", index=False)
print(out.to_string(index=False))
print("\nSaved sofa_candidates.csv with", len(out), "rows")
