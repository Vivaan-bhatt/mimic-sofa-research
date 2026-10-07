import os, sys
import numpy as np
import pandas as pd

base = os.path.expanduser(sys.argv[1])
os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)
MASK_BELOW = 10   # category counts below this are shown as "<10"

icu = pd.read_csv(f"{base}/icu/icustays.csv.gz", parse_dates=["intime", "outtime"])
adm = pd.read_csv(f"{base}/hosp/admissions.csv.gz",
                  usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                           "admission_type", "insurance", "race", "hospital_expire_flag"],
                  parse_dates=["admittime", "dischtime", "deathtime"])
pat = pd.read_csv(f"{base}/hosp/patients.csv.gz",
                  usecols=["subject_id", "gender", "anchor_age", "anchor_year"])

def mask(n):
    return f"<{MASK_BELOW}" if n < MASK_BELOW else f"{n:,}"

flow = []
def log(step, d):
    flow.append({"step": step, "icu_stays": d["stay_id"].nunique(),
                 "hospitalizations": d["hadm_id"].nunique(), "patients": d["subject_id"].nunique()})
    print(f"{step:55s} stays={flow[-1]['icu_stays']:>7,}  hadm={flow[-1]['hospitalizations']:>7,}  patients={flow[-1]['patients']:>7,}")

# --- structural checks (no outcomes involved)
print("== structural checks ==")
print("duplicate stay_id rows:", int(icu["stay_id"].duplicated().sum()))
print("duplicate (subject_id, hadm_id) in admissions:", int(adm.duplicated(["subject_id", "hadm_id"]).sum()))
print("stays with outtime <= intime:", int((icu["outtime"] <= icu["intime"]).sum()))
print("stays with missing intime/outtime:", int(icu[["intime", "outtime"]].isna().any(axis=1).sum()))
print()

df = icu.merge(adm, on=["subject_id", "hadm_id"], how="left", validate="m:1", indicator="adm_merge")
print("stays with no matching admission row:", int((df["adm_merge"] != "both").sum()))
df = df.drop(columns="adm_merge").merge(pat, on="subject_id", how="left", validate="m:1")
df["age"] = df["anchor_age"] + df["intime"].dt.year - df["anchor_year"]

print("== cohort flow ==")
log("1. All MIMIC-IV ICU stays", df)
df = df[df["age"] >= 18]
log("2. Adults (age >= 18 at ICU admission)", df)
df = df.sort_values(["hadm_id", "intime", "stay_id"])
df["icu_seq_in_hadm"] = df.groupby("hadm_id").cumcount() + 1
df = df[df["icu_seq_in_hadm"] == 1]
log("3. First ICU stay per hospitalization", df)
df = df[df["hospital_expire_flag"].isin([0, 1])]
log("4. In-hospital mortality status available", df)
df = df[df["intime"].notna() & df["outtime"].notna() & (df["outtime"] > df["intime"])]
log("5. Valid ICU timestamps", df)

# first stay per patient (sensitivity cohort flag)
df = df.sort_values(["subject_id", "intime"])
df["first_stay_per_patient"] = ~df.duplicated("subject_id", keep="first")

# window and outcomes
df["window_start"] = df["intime"]
df["window_end"] = df["intime"] + pd.Timedelta(hours=24)
df["hospital_death"] = df["hospital_expire_flag"].astype(int)
df["icu_death_strict"] = ((df["hospital_death"] == 1) & df["deathtime"].notna()
                          & (df["deathtime"] >= df["intime"]) & (df["deathtime"] <= df["outtime"])).astype(int)

print("\n== outcome timing (counts only) ==")
hd = df[df["hospital_death"] == 1]
print("in-hospital deaths (all):", f"{len(hd):,}", f"({len(hd)/len(df):.1%} of cohort)")
print("  with no deathtime recorded:", mask(int(hd["deathtime"].isna().sum())))
print("  ICU death by strict definition (deathtime within ICU stay):", f"{int(df['icu_death_strict'].sum()):,}")
after = hd[(hd["deathtime"] > hd["outtime"]) & (hd["deathtime"] <= hd["outtime"] + pd.Timedelta(hours=24))]
print("  hospital deaths within 24h AFTER ICU discharge (tolerance question):", mask(len(after)))
early = hd[hd["deathtime"].notna() & (hd["deathtime"] < hd["window_end"])]
print("  deaths before the 24h window ends (window truncated):", mask(len(early)))
print("stays shorter than 24h:", f"{int(((df['outtime'] - df['intime']) < pd.Timedelta(hours=24)).sum()):,}")
print("hospitalizations per patient > 1 in cohort:", f"{int((~df['first_stay_per_patient']).sum()):,}", "stays are not the patient's first")

# subgroup category counts (NO outcomes shown)
print("\n== subgroup categories (counts only; used to freeze categories) ==")
for col in ["gender", "insurance", "race", "admission_type", "first_careunit"]:
    vc = df[col].fillna("MISSING").value_counts()
    print(f"\n{col} ({len(vc)} categories)")
    for k, v in vc.items():
        print(f"  {str(k):55s} {mask(int(v))}")
print("\nage: median", df["age"].median(), " IQR", df["age"].quantile(.25), "-", df["age"].quantile(.75),
      " range", df["age"].min(), "-", df["age"].max())

keep = ["stay_id", "subject_id", "hadm_id", "intime", "outtime", "window_start", "window_end",
        "age", "gender", "race", "insurance", "admission_type", "first_careunit",
        "hospital_death", "icu_death_strict", "deathtime", "first_stay_per_patient"]
df[keep].to_csv("outputs/tables/cohort_PRIVATE_do_not_share.csv", index=False)
pd.DataFrame(flow).to_csv("outputs/tables/cohort_flow.csv", index=False)
print("\nSaved outputs/tables/cohort_flow.csv (safe to share) and cohort_PRIVATE_do_not_share.csv (stay-level, keep local)")
