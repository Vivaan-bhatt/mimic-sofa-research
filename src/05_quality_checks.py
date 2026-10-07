# 05_quality_checks.py -- outcome-blind audit of SOFA-1 and SOFA-2 inputs and scores.
# Uses scores, measurements and covariates ONLY (no mortality). Aggregate output; small cells masked.
# usage: python3 src/05_quality_checks.py [strict|lookback6] [sofa1_variant]   (sofa1_variant: primary | ett_charted)
import os, sys
import numpy as np
import pandas as pd
from common import *

mode = sys.argv[1] if len(sys.argv) > 1 else "strict"
v1 = sys.argv[2] if len(sys.argv) > 2 else "primary"
tag1 = mode + ("" if v1 == "primary" else "_" + v1)
MASK = int(os.environ.get("MASK", 10))
pd.set_option("display.width", 250, "display.max_columns", 40)

coh = load_cohort()
s1 = pd.read_csv(f"outputs/tables/sofa1_{tag1}_PRIVATE.csv").set_index("stay_id")
s2 = pd.read_csv(f"outputs/tables/sofa2_{mode}_PRIVATE.csv").set_index("stay_id")
cov = coh.set_index("stay_id")
chart = read_extract("chart_long", time_cols=["charttime"], num_cols=["valuenum"])
lab = read_extract("lab_long", time_cols=["charttime"], num_cols=["valuenum"])
N = len(coh)

def m(n):
    return f"<{MASK}" if n < MASK else f"{n:,}"

# ------------------------------------------------------------------ A. brain: intubation and SOFA-1 -> SOFA-2
print(f"== A. Brain: SOFA-1 ({tag1}) vs SOFA-2, by intubation ==")
v = chart[(chart["itemid"] == 223900) & (chart["value"] == "No Response-ETT")]
intub = set(v.loc[in_window(v, coh, mode, time_col="charttime"), "stay_id"])
c1 = s1["cns_pts"].fillna(0).astype(int); c2 = s2["brain_pts"].fillna(0).astype(int)
isint = pd.Series(c1.index.isin(intub), index=c1.index)
print(f"stays with intubated verbal text in window: {m(int(isint.sum()))} ({100*isint.mean():.1f}%)")
for name, mask_ in [("all stays", pd.Series(True, index=c1.index)), ("intubated", isint), ("not intubated", ~isint)]:
    ct = pd.crosstab(c1[mask_], c2[mask_]).reindex(index=range(5), columns=range(5), fill_value=0)
    print(f"\n{name}: rows = SOFA-1 CNS points, columns = SOFA-2 brain points")
    print(ct.map(lambda x: m(int(x))).to_string())
b4 = c2 == 4
print(f"\nSOFA-2 brain = 4: {m(int(b4.sum()))} stays; intubated share {100*(b4 & isint).sum()/max(1,b4.sum()):.1f}%")

# ------------------------------------------------------------------ B. MAP artifact sensitivity
print("\n== B. Minimum MAP in window: sensitivity to the lower plausible bound ==")
mp = chart[chart["itemid"].isin([220052, 220181, 225312])].copy()
mp = mp[in_window(mp, coh, mode, time_col="charttime")]
for lb in [10, 20, 30, 40, 50]:
    v_ = mp["valuenum"].where(mp["valuenum"].between(lb, 300))
    mn = v_.groupby(mp["stay_id"]).min().dropna()
    print(f"  lower bound {lb:>2}: stays with MAP {len(mn):>6,}; MAP<70 in {100*(mn<70).mean():.1f}%; "
          f"median min MAP {mn.median():.0f}; 5th pct {mn.quantile(.05):.0f}")
print(f"  readings per stay in window: median {mp.groupby('stay_id').size().median():.0f}")

# ------------------------------------------------------------------ C. respiratory data funnel
print("\n== C. Respiratory data funnel (stays) ==")
spec = lab[lab["itemid"] == 52033][["stay_id", "specimen_id", "value"]].drop_duplicates(["stay_id", "specimen_id"])
pa = lab[lab["itemid"] == 50821].merge(spec.rename(columns={"value": "spec"}), on=["stay_id", "specimen_id"], how="left")
pa = pa[(pa["spec"].fillna("").str.strip() == "ART.") & pa["valuenum"].between(*RANGES["pao2"]) & in_window(pa, coh, mode, time_col="charttime")]
n_art = pa["stay_id"].nunique()
fio2c = chart[(chart["itemid"] == 223835) & in_window(chart, coh, mode, time_col="charttime")]
spo2 = chart[(chart["itemid"] == 220277) & in_window(chart, coh, mode, time_col="charttime")]
print(f"  arterial PaO2 in window:                       {m(n_art)} ({100*n_art/N:.1f}%)")
print(f"  of those, P/F computed in SOFA-1 (FiO2 found): {m(int(s1['pf_min'].notna().sum()))}")
print(f"  FiO2 charted in window:                        {m(fio2c['stay_id'].nunique())}")
print(f"  SpO2 charted in window:                        {m(spo2['stay_id'].nunique())}")
print(f"  SOFA-1 respiratory observed: {m(int(s1['resp_pts'].notna().sum()))}; SOFA-2: {m(int(s2['resp_pts'].notna().sum()))}")
print(f"  SOFA-2 respiratory from SpO2/FiO2 fallback only: {m(int((s2['resp_sf_pts'].notna() & s2['resp_pf_pts'].isna()).sum()))}")

# ------------------------------------------------------------------ D/E. subgroup tables (no outcomes)
def race_group(r):
    r = str(r).upper()
    if r in ("UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"):
        return "Unknown/Declined"
    for pref, g in [("WHITE", "White"), ("BLACK", "Black"), ("HISPANIC", "Hispanic/Latino"), ("ASIAN", "Asian")]:
        if r.startswith(pref):
            return g
    return "Other"            # includes OTHER, PORTUGUESE, SOUTH AMERICAN, AI/AN, NHPI, MULTIPLE
ins = cov["insurance"].fillna("Missing")
groups = {"sex": cov["gender"].map({"M": "Male", "F": "Female"}).fillna("Unknown"),
          "race": cov["race"].map(race_group),
          "insurance": ins.where(ins.isin(["Medicare", "Medicaid", "Private"]), "Other/Missing")}
d1 = {"resp_pts": "resp", "coag_pts": "hemo", "liver_pts": "liver", "cv_pts": "cv", "cns_pts": "brain", "renal_pts": "renal"}
d2 = {"resp_pts": "resp", "hemo_pts": "hemo", "liver_pts": "liver", "cv_pts": "cv", "brain_pts": "brain", "renal_pts": "renal"}
miss_rows, delta_rows = [], []
print("\n== D. Percent of stays with each domain OBSERVED, by subgroup (SOFA-1 / SOFA-2) ==")
for gname, g in groups.items():
    rows = []
    for lvl in g.value_counts().index:
        idx = g.index[g == lvl]
        if len(idx) < MASK:
            continue
        r = {"group": gname, "level": lvl, "n": len(idx)}
        for k, nm in d1.items():
            r[f"s1_{nm}"] = round(100 * s1.loc[idx, k].notna().mean(), 1)
        for k, nm in d2.items():
            r[f"s2_{nm}"] = round(100 * s2.loc[idx, k].notna().mean(), 1)
        r["s1_complete"] = round(100 * s1.loc[idx, "complete_case"].mean(), 1)
        r["s2_complete"] = round(100 * s2.loc[idx, "complete_case"].mean(), 1)
        r["mean_sofa1"] = round(s1.loc[idx, "sofa1_total"].mean(), 2)
        r["mean_sofa2"] = round(s2.loc[idx, "sofa2_total"].mean(), 2)
        rows.append(r)
        dd = {"group": gname, "level": lvl, "n": len(idx)}
        for (k1, nm), (k2, _) in zip(d1.items(), d2.items()):
            dd[f"d_{nm}"] = round(s2.loc[idx, k2].fillna(0).mean() - s1.loc[idx, k1].fillna(0).mean(), 3)
        dd["d_total"] = round(s2.loc[idx, "sofa2_total"].mean() - s1.loc[idx, "sofa1_total"].mean(), 3)
        delta_rows.append(dd)
    t = pd.DataFrame(rows); miss_rows.append(t)
    print(f"\n{gname}"); print(t.drop(columns="group").to_string(index=False))
miss = pd.concat(miss_rows); miss.to_csv("outputs/tables/audit_missingness_by_subgroup.csv", index=False)
print("\n== E. Mean change in points per domain (SOFA-2 minus SOFA-1, missing = 0), by subgroup ==")
dl = pd.DataFrame(delta_rows); dl.to_csv("outputs/tables/audit_delta_by_subgroup.csv", index=False)
for gname in groups:
    print(f"\n{gname}"); print(dl[dl.group == gname].drop(columns="group").to_string(index=False))
print("\nSaved outputs/tables/audit_missingness_by_subgroup.csv and audit_delta_by_subgroup.csv (aggregates only)")
