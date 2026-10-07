# 02_extract.py -- pull raw measurements for the analytic cohort into local files.
# Window kept: [intime - 6h, intime + 24h]; scoring scripts apply the exact protocol window
# (primary: [intime, intime+24h); sensitivity: with 6h lookback). Outputs are PATIENT-LEVEL: keep local.
import os, sys, gzip, json, time
import numpy as np
import pandas as pd

base = os.path.expanduser(sys.argv[1])
OUT = "outputs/extract"
os.makedirs(OUT, exist_ok=True)
CHUNK = 5_000_000
READ = dict(dtype=str, na_filter=False)   # na_filter=False keeps text like "None" and "NA" intact

coh = pd.read_csv("outputs/tables/cohort_PRIVATE_do_not_share.csv",
                  parse_dates=["intime", "outtime", "window_end"])
coh["lo"] = coh["intime"] - pd.Timedelta(hours=6)
coh["hi"] = coh["window_end"]
coh["hi_weight"] = coh["intime"] + pd.Timedelta(hours=72)
stay_lo = coh.set_index("stay_id")["lo"]
stay_hi = coh.set_index("stay_id")["hi"]
stay_hiw = coh.set_index("stay_id")["hi_weight"]
stay_set = set(coh["stay_id"].astype(str))
subj_set = set(coh["subject_id"].astype(str))
hadm_set = set(coh["hadm_id"].astype(str))
coh_s = coh[["subject_id", "hadm_id", "stay_id", "lo", "hi"]]

handles, first, counts, stays = {}, {}, {}, {}
def append(df, name):
    if df.empty:
        return
    if name not in handles:
        handles[name] = gzip.open(f"{OUT}/{name}.csv.gz", "wt")
        first[name], counts[name], stays[name] = True, 0, set()
    df.to_csv(handles[name], header=first[name], index=False)
    first[name] = False
    counts[name] += len(df)
    stays[name].update(df["stay_id"].unique())

def stay_table(path, items, usecols, name, point_col=None, weight_items=()):
    """Stay-keyed table. point_col -> point events; otherwise interval events (starttime/endtime)."""
    items = set(str(i) for i in items)
    weight_items = set(str(i) for i in weight_items)
    t0 = time.time(); n = 0
    for k, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=CHUNK, **READ)):
        n += len(chunk)
        sub = chunk[chunk["itemid"].isin(items) & chunk["stay_id"].isin(stay_set)]
        if len(sub):
            sid = sub["stay_id"].astype("int64")
            lo = sid.map(stay_lo)
            hi = sid.map(stay_hi)
            if weight_items:
                hi = hi.where(~sub["itemid"].isin(weight_items), sid.map(stay_hiw))
            if point_col:
                t = pd.to_datetime(sub[point_col])
                keep = (t >= lo) & (t <= hi)
            else:
                s, e = pd.to_datetime(sub["starttime"]), pd.to_datetime(sub["endtime"])
                keep = (s < hi) & (e > lo)
            sub = sub[keep].copy()
            sub["stay_id"] = sub["stay_id"].astype("int64")
            append(sub, name)
        if k % 10 == 0:
            print(f"  {name}: {n:,} rows read ({time.time()-t0:,.0f}s)", flush=True)
    print(f"{name}: kept {counts.get(name, 0):,} rows from {len(stays.get(name, [])):,} stays", flush=True)

CHART = [220739, 223900, 223901,            # GCS eye / verbal / motor
         223835, 220277,                    # FiO2, SpO2
         220052, 220181, 225312,            # MAP
         223849, 229314, 226732, 225949, 227577, 227583,   # ventilator mode, O2 device, NIV/CPAP
         229270, 224660, 228193, 229268,    # ECMO flow / oxygenator / circuit configuration
         227980, 228154, 229671, 220125, 220128, 229254,   # IABP, Impella, VADs
         226512, 224639, 226531,            # weights
         228096,                            # RASS
         225126, 227290]                    # dialysis patient flag, CRRT mode
WEIGHTS = [226512, 224639, 226531]
LAB = [50885, 53089, 50912, 52546, 51265, 53189, 50821, 50816, 52033, 50820, 50882, 50971, 52610]
INPUT = [221906, 221289, 221662, 221653, 221749, 229630, 229632, 222315, 221986, 229709, 229764,   # vasoactive
         222168, 221668, 225150, 229420, 221385, 221712, 221744, 225942,                           # sedation / opioids
         221824]                                                                                   # haloperidol
PROC = [225792, 225794, 225441, 225802, 225805, 229529, 229530, 224272, 228169]
OUTP = [226559, 226560, 226561, 226563, 226564, 226565, 226567, 226584, 226557, 226558, 227488, 227489, 226627, 226631]

print("== chartevents"); stay_table(f"{base}/icu/chartevents.csv.gz", CHART,
    ["stay_id", "itemid", "charttime", "value", "valuenum", "valueuom"], "chart_long", point_col="charttime", weight_items=WEIGHTS)
print("== inputevents"); stay_table(f"{base}/icu/inputevents.csv.gz", INPUT,
    ["stay_id", "itemid", "starttime", "endtime", "amount", "amountuom", "rate", "rateuom",
     "ordercategorydescription", "patientweight", "statusdescription"], "input_long")
print("== procedureevents"); stay_table(f"{base}/icu/procedureevents.csv.gz", PROC,
    ["stay_id", "itemid", "starttime", "endtime", "value", "valueuom", "statusdescription"], "proc_long")
print("== outputevents"); stay_table(f"{base}/icu/outputevents.csv.gz", OUTP,
    ["stay_id", "itemid", "charttime", "value", "valueuom"], "output_long", point_col="charttime")

print("== labevents (joined to stays by subject_id and time)")
items = set(str(i) for i in LAB); t0 = time.time(); n = 0
for k, chunk in enumerate(pd.read_csv(f"{base}/hosp/labevents.csv.gz",
        usecols=["subject_id", "specimen_id", "itemid", "charttime", "value", "valuenum", "valueuom"],
        chunksize=CHUNK, **READ)):
    n += len(chunk)
    sub = chunk[chunk["itemid"].isin(items) & chunk["subject_id"].isin(subj_set)].copy()
    if len(sub):
        sub["subject_id"] = sub["subject_id"].astype("int64")
        m = sub.merge(coh_s[["subject_id", "stay_id", "lo", "hi"]], on="subject_id")
        t = pd.to_datetime(m["charttime"])
        m = m[(t >= m["lo"]) & (t <= m["hi"])]
        append(m[["stay_id", "specimen_id", "itemid", "charttime", "value", "valuenum", "valueuom"]], "lab_long")
    if k % 10 == 0:
        print(f"  lab_long: {n:,} rows read ({time.time()-t0:,.0f}s)", flush=True)
print(f"lab_long: kept {counts.get('lab_long', 0):,} rows from {len(stays.get('lab_long', [])):,} stays")

print("== prescriptions (antipsychotics / dexmedetomidine overlapping the window)")
pat = r"haloperidol|quetiapine|olanzapine|risperidone|ziprasidone|aripiprazole|chlorpromazine|dexmedetomidine"
t0 = time.time(); n = 0
for k, chunk in enumerate(pd.read_csv(f"{base}/hosp/prescriptions.csv.gz",
        usecols=["hadm_id", "drug", "route", "starttime", "stoptime"], chunksize=CHUNK, **READ)):
    n += len(chunk)
    sub = chunk[chunk["hadm_id"].isin(hadm_set) & chunk["drug"].str.lower().str.contains(pat, regex=True)].copy()
    if len(sub):
        sub["hadm_id"] = sub["hadm_id"].astype("int64")
        m = sub.merge(coh_s[["hadm_id", "stay_id", "lo", "hi"]], on="hadm_id")
        s = pd.to_datetime(m["starttime"], errors="coerce"); e = pd.to_datetime(m["stoptime"], errors="coerce")
        m = m[(s <= m["hi"]) & (e.isna() | (e >= m["lo"]))]
        append(m[["stay_id", "hadm_id", "drug", "route", "starttime", "stoptime"]], "rx_long")
    if k % 10 == 0:
        print(f"  rx_long: {n:,} rows read ({time.time()-t0:,.0f}s)", flush=True)
print(f"rx_long: kept {counts.get('rx_long', 0):,} rows from {len(stays.get('rx_long', [])):,} stays")

print("== diagnoses (chronic dialysis / ESRD codes)")
CODES = {("9", "5856"), ("9", "V451"), ("9", "V4511"), ("9", "V4512"), ("10", "N186"), ("10", "Z992")}
dx = pd.read_csv(f"{base}/hosp/diagnoses_icd.csv.gz", usecols=["hadm_id", "icd_code", "icd_version"], **READ)
dx = dx[dx["hadm_id"].isin(hadm_set)]
dx = dx[[(v, c.strip()) in CODES for v, c in zip(dx["icd_version"], dx["icd_code"])]].copy()
dx["hadm_id"] = dx["hadm_id"].astype("int64")
dd = dx.merge(coh[["hadm_id", "stay_id"]], on="hadm_id")[["stay_id", "hadm_id", "icd_code", "icd_version"]]
append(dd, "dialysis_dx")
print(f"dialysis_dx: {counts.get('dialysis_dx', 0):,} rows from {len(stays.get('dialysis_dx', [])):,} stays")

for h in handles.values():
    h.close()
manifest = {name: {"rows": counts[name], "stays": len(stays[name])} for name in counts}
json.dump(manifest, open("outputs/tables/extract_manifest.json", "w"), indent=2)
print("\nSaved extract files in outputs/extract/ (patient-level, keep local) and outputs/tables/extract_manifest.json")
