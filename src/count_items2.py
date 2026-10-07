import os, sys
import pandas as pd
from collections import Counter, defaultdict

base = os.path.expanduser(sys.argv[1])
MIN_COUNT = 20
CHUNK = 5_000_000

def show(counter, n=25):
    for v, c in counter.most_common(n):
        if c >= MIN_COUNT:
            print(f"    {str(v)[:90]:90s} {c:>10,}")

# ---- A. chartevents text values (GCS verbal/motor/eye text, ECMO circuit configuration)
ids = {"223900": "GCS - Verbal Response", "223901": "GCS - Motor Response",
       "220739": "GCS - Eye Opening", "229268": "Circuit Configuration (ECMO)"}
cnt = defaultdict(Counter)
for chunk in pd.read_csv(f"{base}/icu/chartevents.csv.gz", usecols=["itemid", "value"],
                         dtype=str, chunksize=CHUNK):
    sub = chunk[chunk["itemid"].isin(ids)]
    for iid, g in sub.groupby("itemid"):
        cnt[iid].update(g["value"].fillna("NA").value_counts().to_dict())
print("== A. chartevents text values ==")
for iid, name in ids.items():
    print(f"\n  {iid} {name}  ({sum(cnt[iid].values()):,} rows)")
    show(cnt[iid])

# ---- B. prescriptions: antipsychotics / delirium-type drugs (row and hospitalization counts)
pat = r"haloperidol|quetiapine|olanzapine|risperidone|ziprasidone|aripiprazole|chlorpromazine|dexmedetomidine"
rows, hadm = Counter(), defaultdict(set)
for chunk in pd.read_csv(f"{base}/hosp/prescriptions.csv.gz", usecols=["hadm_id", "drug", "route"],
                         dtype=str, chunksize=CHUNK):
    d = chunk["drug"].fillna("").str.lower()
    sub = chunk[d.str.contains(pat, regex=True)]
    for (drug, route), g in sub.groupby([sub["drug"].str.lower(), sub["route"].fillna("NA")]):
        rows[(drug, route)] += len(g)
        hadm[(drug, route)].update(g["hadm_id"].dropna().unique())
print("\n== B. prescriptions (drug, route): rows / hospitalizations ==")
for (drug, route), c in rows.most_common(40):
    if c >= MIN_COUNT:
        print(f"    {drug[:45]:45s} {route[:12]:12s} {c:>9,} rows {len(hadm[(drug, route)]):>8,} hadm")

# ---- C. chronic dialysis / ESRD diagnosis codes
dx = pd.read_csv(f"{base}/hosp/diagnoses_icd.csv.gz", usecols=["hadm_id", "icd_code", "icd_version"], dtype=str)
codes = {("9", "5856"), ("9", "V451"), ("9", "V4511"), ("9", "V4512"), ("9", "V560"), ("9", "V561"),
         ("9", "V562"), ("9", "V568"), ("10", "N186"), ("10", "Z992"), ("10", "Z4901"), ("10", "Z4902")}
mask = [(v, c) in codes for v, c in zip(dx["icd_version"], dx["icd_code"].str.strip())]
sub = dx[mask]
titles = pd.read_csv(f"{base}/hosp/d_icd_diagnoses.csv.gz", dtype=str)
titles["icd_code"] = titles["icd_code"].str.strip()
res = (sub.assign(icd_code=sub["icd_code"].str.strip())
          .groupby(["icd_version", "icd_code"])["hadm_id"].nunique().reset_index(name="n_hospitalizations")
          .merge(titles, on=["icd_version", "icd_code"], how="left"))
print("\n== C. dialysis / ESRD diagnosis codes (hospitalizations) ==")
print(res.to_string(index=False))
