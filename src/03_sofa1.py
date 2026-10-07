# 03_sofa1.py -- SOFA-1 (Vincent 1996), independent calculator on the shared cleaned inputs.
# Conventions follow the MIT-LCP mimic-code reference where the 1996 paper is silent
# (documented in the protocol): intubated patients (verbal text "No Response-ETT") are scored GCS 15;
# respiratory support = invasive ventilation; UO used only if the stay covers >= 22 h; missing = 0 points.
import sys
import numpy as np
import pandas as pd
from common import *

mode = sys.argv[1] if len(sys.argv) > 1 else "strict"
variant = sys.argv[2] if len(sys.argv) > 2 else "primary"   # primary | ett_charted
tag = mode + ("" if variant == "primary" else "_" + variant)
coh = load_cohort()
base = coh[["stay_id", "intime", "outtime"]].copy()
stay_hours = (coh.set_index("stay_id")["outtime"] - coh.set_index("stay_id")["intime"]) / H

res = pd.DataFrame({"stay_id": coh["stay_id"]}).set_index("stay_id")

# ---------------- CNS (GCS) ----------------
chart = read_extract("chart_long", time_cols=["charttime"], num_cols=["valuenum"])
g = chart[chart["itemid"].isin([220739, 223900, 223901])].copy()
g["comp"] = g["itemid"].map({220739: "eye", 223900: "verbal", 223901: "motor"})
g["val"] = np.where((g["itemid"] == 223900) & (g["value"] == "No Response-ETT"), 0, g["valuenum"])
for comp, key in [("eye", "gcs_eye"), ("verbal", "gcs_verbal"), ("motor", "gcs_motor")]:
    m = g["comp"] == comp
    g.loc[m, "val"] = rng(g.loc[m, "val"], key)
piv = (g.pivot_table(index=["stay_id", "charttime"], columns="comp", values="val", aggfunc="max")
         .reset_index().sort_values(["stay_id", "charttime"]).reset_index(drop=True))
for comp in ["eye", "verbal", "motor"]:
    if comp not in piv:
        piv[comp] = np.nan
    t_last = piv["charttime"].where(piv[comp].notna()).groupby(piv["stay_id"]).ffill()
    fwd = piv[comp].groupby(piv["stay_id"]).ffill()
    piv[comp + "_f"] = fwd.where((piv["charttime"] - t_last) <= 6 * H)
vf = piv["verbal_f"].where(piv["verbal_f"] != 0, 1) if variant == "ett_charted" else piv["verbal_f"]   # ETT verbal scored as charted (1)
piv["gcs"] = np.where(vf == 0, 15, piv["motor_f"].fillna(6) + vf.fillna(5) + piv["eye_f"].fillna(4))
has_any = piv[["eye", "verbal", "motor"]].notna().any(axis=1)
piv = piv[has_any & in_window(piv, coh, mode, time_col="charttime")]
res["gcs_min"] = piv.groupby("stay_id")["gcs"].min()

# ---------------- MAP (cardiovascular, part 1) ----------------
mp = chart[chart["itemid"].isin([220052, 220181, 225312])].copy()
mp["v"] = rng(mp["valuenum"], "map")
mp = mp[mp["v"].notna() & in_window(mp, coh, mode, time_col="charttime")]
res["map_min"] = mp.groupby("stay_id")["v"].min()

# ---------------- labs ----------------
lab = read_extract("lab_long", time_cols=["charttime"], num_cols=["valuenum"])
def lab_extreme(items, key, fn):
    d = lab[lab["itemid"].isin(items)].copy()
    d["v"] = rng(d["valuenum"], key)
    d = d[d["v"].notna() & in_window(d, coh, mode, time_col="charttime")]
    return d.groupby("stay_id")["v"].agg(fn)
res["platelets_min"] = lab_extreme([51265, 53189], "platelets", "min")
res["bilirubin_max"] = lab_extreme([50885, 53089], "bilirubin", "max")
res["creatinine_max"] = lab_extreme([50912, 52546], "creatinine", "max")

# ---------------- respiratory: arterial PaO2/FiO2 + invasive ventilation ----------------
spec = lab[lab["itemid"] == 52033][["stay_id", "specimen_id", "value"]].rename(columns={"value": "spec_type"})
spec = spec.drop_duplicates(["stay_id", "specimen_id"])
pao2 = lab[lab["itemid"] == 50821][["stay_id", "specimen_id", "charttime", "valuenum"]].rename(columns={"valuenum": "pao2"})
pao2["pao2"] = rng(pao2["pao2"], "pao2")
pao2 = pao2.merge(spec, on=["stay_id", "specimen_id"], how="left")
pao2 = pao2[pao2["spec_type"].fillna("").str.strip() == "ART."]
fl = lab[lab["itemid"] == 50816][["stay_id", "specimen_id", "valuenum"]].rename(columns={"valuenum": "fio2_lab"})
fl["fio2_lab"] = fio2_percent(fl["fio2_lab"])
pao2 = pao2.merge(fl.drop_duplicates(["stay_id", "specimen_id"]), on=["stay_id", "specimen_id"], how="left")
fc = chart[chart["itemid"] == 223835][["stay_id", "charttime", "valuenum"]].rename(columns={"valuenum": "fio2_chart"})
fc["fio2_chart"] = fio2_percent(fc["fio2_chart"])
fc = fc[fc["fio2_chart"].notna()].sort_values("charttime")
pao2 = pao2.sort_values("charttime")
pao2 = pd.merge_asof(pao2, fc, on="charttime", by="stay_id", direction="backward", tolerance=4 * H)
pao2["fio2"] = pao2["fio2_lab"].fillna(pao2["fio2_chart"])
pao2 = pao2[pao2["pao2"].notna() & pao2["fio2"].notna()]
pao2["pf"] = pao2["pao2"] / (pao2["fio2"] / 100.0)
proc = read_extract("proc_long", time_cols=["starttime", "endtime"])
vent = proc[proc["itemid"] == 225792][["stay_id", "starttime", "endtime"]].sort_values("starttime")
pao2 = pd.merge_asof(pao2.sort_values("charttime"), vent, left_on="charttime", right_on="starttime",
                     by="stay_id", direction="backward")
pao2["vent"] = pao2["endtime"].notna() & (pao2["charttime"] <= pao2["endtime"])
pao2 = pao2[in_window(pao2, coh, mode, time_col="charttime")]
pao2["pts"] = np.select(
    [pao2["vent"] & (pao2["pf"] < 100), pao2["vent"] & (pao2["pf"] < 200),
     pao2["pf"] < 300, pao2["pf"] < 400], [4, 3, 2, 1], default=0)
res["pf_min"] = pao2.groupby("stay_id")["pf"].min()
res["resp_pts"] = pao2.groupby("stay_id")["pts"].max()

# ---------------- cardiovascular: dopamine, dobutamine, epinephrine, norepinephrine ----------------
inp = read_extract("input_long", time_cols=["starttime", "endtime"], num_cols=["rate"])
inp = inp[(inp["rateuom"] == "mcg/kg/min") & (inp["statusdescription"] != "Paused")
          & inp["itemid"].isin([221662, 221653, 221289, 221906]) & (inp["rate"] > 0)]
inp = inp[in_window(inp, coh, mode, start_col="starttime", end_col="endtime")]
drug = {221662: "dopamine", 221653: "dobutamine", 221289: "epinephrine", 221906: "norepinephrine"}
inp["drug"] = inp["itemid"].map(drug)
inp["valid_rate"] = np.nan
for dname in ["dopamine", "epinephrine", "norepinephrine"]:
    m = inp["drug"] == dname
    inp.loc[m, "valid_rate"] = rng(inp.loc[m, "rate"], dname)
on = inp.groupby(["stay_id", "drug"]).size().unstack(fill_value=0) > 0          # any infusion (dose possibly unknown)
mx = inp.groupby(["stay_id", "drug"])["valid_rate"].max().unstack()
for dname in drug.values():
    res[dname + "_on"] = on[dname] if dname in on else False
    res[dname + "_max"] = mx[dname] if dname in mx else np.nan
for dname in drug.values():
    res[dname + "_on"] = res[dname + "_on"].astype("boolean").fillna(False).astype(bool)

# ---------------- renal: creatinine and 24 h urine output ----------------
out = read_extract("output_long", time_cols=["charttime"], num_cols=["value"])
uo_items = [226559, 226560, 226561, 226563, 226564, 226565, 226567, 226584, 226557, 226558, 227489, 227488]
out = out[out["itemid"].isin(uo_items)].copy()
out["v"] = rng(out["value"], "uo_record")
out.loc[out["itemid"] == 227488, "v"] = -out.loc[out["itemid"] == 227488, "v"]     # irrigant in = negative
out = out[out["v"].notna() & in_window(out, coh, mode, time_col="charttime")]
uo_sum = out.groupby("stay_id")["v"].sum()
cov = stay_hours.clip(upper=24)
res["uo_24h"] = (uo_sum * 24 / cov.reindex(uo_sum.index)).where(cov.reindex(uo_sum.index) >= 22)

# ---------------- points ----------------
r = res
r["coag_pts"] = np.select([r["platelets_min"] < 20, r["platelets_min"] < 50, r["platelets_min"] < 100,
                           r["platelets_min"] < 150], [4, 3, 2, 1], default=np.where(r["platelets_min"].isna(), np.nan, 0))
r["liver_pts"] = np.select([r["bilirubin_max"] >= 12, r["bilirubin_max"] >= 6, r["bilirubin_max"] >= 2,
                            r["bilirubin_max"] >= 1.2], [4, 3, 2, 1], default=np.where(r["bilirubin_max"].isna(), np.nan, 0))
r["cns_pts"] = np.select([r["gcs_min"] < 6, r["gcs_min"] < 10, r["gcs_min"] < 13, r["gcs_min"] < 15],
                         [4, 3, 2, 1], default=np.where(r["gcs_min"].isna(), np.nan, 0))
cv4 = (r["dopamine_max"] > 15) | (r["epinephrine_max"] > 0.1) | (r["norepinephrine_max"] > 0.1)
cv3 = (r["dopamine_max"] > 5) | r["epinephrine_on"] | r["norepinephrine_on"]
cv2 = r["dopamine_on"] | r["dobutamine_on"]
cv1 = r["map_min"] < 70
r["cv_pts"] = np.select([cv4, cv3, cv2, cv1], [4, 3, 2, 1],
                        default=np.where(r["map_min"].isna() & ~(r["dopamine_on"] | r["dobutamine_on"] | r["epinephrine_on"] | r["norepinephrine_on"]), np.nan, 0))
c = r["creatinine_max"]; u = r["uo_24h"]
r["renal_pts"] = np.select([(c >= 5.0) | (u < 200), (c >= 3.5) | (u < 500), c >= 2.0, c >= 1.2],
                           [4, 3, 2, 1], default=np.where(c.isna() & u.isna(), np.nan, 0))
dom = ["resp_pts", "coag_pts", "liver_pts", "cv_pts", "cns_pts", "renal_pts"]
r["n_domains_observed"] = r[dom].notna().sum(axis=1)
r["complete_case"] = r["n_domains_observed"] == 6
r["sofa1_total"] = r[dom].fillna(0).sum(axis=1).astype(int)

r.reset_index().to_csv(f"outputs/tables/sofa1_{tag}_PRIVATE.csv", index=False)

# ---------------- aggregate summary (safe to share) ----------------
n = len(r)
print(f"SOFA-1 ({tag})  N stays = {n:,}")
rows = []
for d in dom:
    vc = r[d].value_counts().reindex([0, 1, 2, 3, 4], fill_value=0)
    obs = int(r[d].notna().sum())
    rows.append({"domain": d, "n_observed": obs, "pct_missing": round(100 * (1 - obs / n), 1),
                 **{f"pts_{k}": int(v) for k, v in vc.items()}})
summ = pd.DataFrame(rows)
print(summ.to_string(index=False))
t = r["sofa1_total"]
print(f"\nTotal SOFA-1 (missing = 0): median {t.median():.0f} [IQR {t.quantile(.25):.0f}, {t.quantile(.75):.0f}], "
      f"mean {t.mean():.2f}, max {t.max()}")
print(f"complete-case stays (all six domains observed): {int(r['complete_case'].sum()):,}")
tc = t[r["complete_case"]]
if len(tc):
    print(f"complete-case total: median {tc.median():.0f} [IQR {tc.quantile(.25):.0f}, {tc.quantile(.75):.0f}]")
dist = t.value_counts().sort_index()
dist.rename_axis("sofa1_total").rename("n").reset_index().to_csv(f"outputs/tables/sofa1_{tag}_distribution.csv", index=False)
summ.to_csv(f"outputs/tables/sofa1_{tag}_domain_summary.csv", index=False)
print(f"\nSaved sofa1_{tag}_PRIVATE.csv (stay-level, keep local) and aggregate summaries in outputs/tables/")
