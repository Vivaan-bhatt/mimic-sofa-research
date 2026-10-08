# 04_sofa2.py -- SOFA-2 (Ranzani et al., JAMA 2025, Table 2), built independently from raw extracts.
# Same cleaned inputs, window and plausible ranges as SOFA-1 (common.py). Rules and operational choices
# are listed in configs/analysis_protocol.yaml (operational_rules).
# usage: python3 src/04_sofa2.py [strict|lookback6] [primary|ne_half|sed_broad]
import sys
import numpy as np
import pandas as pd
from common import *

mode = sys.argv[1] if len(sys.argv) > 1 else "strict"
variant = sys.argv[2] if len(sys.argv) > 2 else "primary"
SED_BROAD = variant == "sed_broad"                     # sensitivity: broader sedation definition for the brain domain
NE_FACTOR = 0.5 if variant == "ne_half" else 1.0       # sensitivity: norepinephrine salt -> base
tag = mode + ("" if variant == "primary" else "_" + variant)

coh = load_cohort()
sidx = coh.set_index("stay_id")
lo_h, hi_h = WINDOWS[mode]
WLO = sidx["intime"] + lo_h * H
WHI = sidx["intime"] + hi_h * H
stay_hours = (sidx["outtime"] - sidx["intime"]) / H
res = pd.DataFrame(index=coh["stay_id"]); res.index.name = "stay_id"

chart = read_extract("chart_long", time_cols=["charttime"], num_cols=["valuenum"])
lab = read_extract("lab_long", time_cols=["charttime"], num_cols=["valuenum"])
inp = read_extract("input_long", time_cols=["starttime", "endtime"], num_cols=["rate"])
proc = read_extract("proc_long", time_cols=["starttime", "endtime"])
outp = read_extract("output_long", time_cols=["charttime"], num_cols=["value"])
rx = read_extract("rx_long", int_cols=("stay_id", "hadm_id"))
dxd = read_extract("dialysis_dx", int_cols=("stay_id", "hadm_id"))

def cin(df, col="charttime"):                                   # point events inside the window
    return in_window(df, coh, mode, time_col=col)

# ============================ labs: hemostasis, liver, kidney/RRT criteria ============================
def lab_extreme(items, key, fn):
    d = lab[lab["itemid"].isin(items)].copy()
    d["v"] = rng(d["valuenum"], key)
    d = d[d["v"].notna() & cin(d)]
    return d.groupby("stay_id")["v"].agg(fn)
res["platelets_min"] = lab_extreme([51265, 53189], "platelets", "min")
res["bilirubin_max"] = lab_extreme([50885, 53089], "bilirubin", "max")
res["creatinine_max"] = lab_extreme([50912, 52546], "creatinine", "max")
res["potassium_max"] = lab_extreme([50971, 52610], "potassium", "max")
res["ph_min"] = lab_extreme([50820], "ph", "min")
res["bicarbonate_min"] = lab_extreme([50882], "bicarbonate", "min")
p = res["platelets_min"]
res["hemo_pts"] = np.select([p <= 50, p <= 80, p <= 100, p <= 150], [4, 3, 2, 1], default=np.where(p.isna(), np.nan, 0))
b = res["bilirubin_max"]
res["liver_pts"] = np.select([b > 12, b > 6, b > 3, b > 1.2], [4, 3, 2, 1], default=np.where(b.isna(), np.nan, 0))

# ============================ brain ============================
piv = gcs_pivot(chart)
total = piv["motor_f"].fillna(6) + piv["verbal_f"].fillna(5) + piv["eye_f"].fillna(4)
pts_total = np.select([total <= 5, total <= 8, total <= 12, total <= 14], [4, 3, 2, 1], default=0)
mo = piv["motor_f"].fillna(6)
pts_motor = np.select([mo <= 2, mo == 3, mo == 4, mo == 5], [4, 3, 2, 1], default=0)   # proposed mapping (footnote b)
piv["pts"] = np.where(piv["verbal_f"] == 0, pts_motor, pts_total)                       # intubated -> motor only

SED = [222168, 221668, 225150, 229420, 221385, 221712]
sed = inp[inp["itemid"].isin(SED) & (inp["ordercategorydescription"] == "Continuous Med")
          & (inp["statusdescription"] != "Paused") & (inp["rate"] > 0)]
sed = sed[["stay_id", "starttime", "endtime"]]
if SED_BROAD:        # adds fentanyl infusions, sedative/opioid pushes (2 h), and RASS <= -3 (4 h)
    cont = inp[inp["itemid"].isin([221744, 225942]) & (inp["ordercategorydescription"] == "Continuous Med")
               & (inp["statusdescription"] != "Paused") & (inp["rate"] > 0)][["stay_id", "starttime", "endtime"]]
    push = inp[inp["itemid"].isin([221668, 221385, 221744, 221712]) & (inp["ordercategorydescription"] == "Drug Push")][["stay_id", "starttime"]].copy()
    push["endtime"] = push["starttime"] + pd.Timedelta(hours=2)
    rass = chart[(chart["itemid"] == 228096) & (chart["valuenum"] <= -3)][["stay_id", "charttime"]].rename(columns={"charttime": "starttime"}).copy()
    rass["endtime"] = rass["starttime"] + pd.Timedelta(hours=4)
    sed = pd.concat([sed, cont, push, rass], ignore_index=True)
if len(sed):
    u = add_union(sed[["stay_id", "starttime", "endtime"]], ["stay_id"])[["stay_id", "gs", "ge"]].drop_duplicates().sort_values("gs")
    piv = pd.merge_asof(piv.sort_values("charttime"), u, left_on="charttime", right_on="gs", by="stay_id", direction="backward")
    piv["sedated"] = piv["ge"].notna() & (piv["charttime"] <= piv["ge"])
else:
    piv["gs"] = pd.NaT; piv["sedated"] = False
piv = piv.reset_index(drop=True); piv["rid"] = piv.index
non = (piv.loc[~piv["sedated"], ["stay_id", "charttime", "pts"]]
          .rename(columns={"charttime": "pre_t", "pts": "pre_pts"}).sort_values("pre_t"))
sr = piv.loc[piv["sedated"], ["rid", "stay_id", "gs"]].sort_values("gs")
if len(sr):
    sr = pd.merge_asof(sr, non, left_on="gs", right_on="pre_t", by="stay_id", direction="backward", allow_exact_matches=False)
    piv = piv.merge(sr[["rid", "pre_pts"]], on="rid", how="left")
else:
    piv["pre_pts"] = np.nan
piv["pts_eff"] = np.where(piv["sedated"], piv["pre_pts"].fillna(0), piv["pts"])      # unknown pre-sedation GCS -> 0
pw = piv[cin(piv)]
res["brain_gcs_pts"] = pw.groupby("stay_id")["pts_eff"].max()

# delirium drugs (antipsychotics; indication is not recorded in MIMIC)
r = rx.copy()
d = r["drug"].str.lower()
r = r[d.str.contains("haloperidol|quetiapine|olanzapine|risperidone|ziprasidone", regex=True)
      & ~d.str.contains("decanoate|long.acting", regex=True)].copy()
r["starttime"] = pd.to_datetime(r["starttime"], errors="coerce")
r["endtime"] = pd.to_datetime(r["stoptime"], errors="coerce")
r["endtime"] = r["endtime"].fillna(r["starttime"] + pd.Timedelta(days=365))
r = r[r["starttime"].notna()]
rxs = set(r.loc[in_window(r, coh, mode, start_col="starttime", end_col="endtime"), "stay_id"])
hal = inp[(inp["itemid"] == 221824)]
rxs |= set(hal.loc[in_window(hal, coh, mode, start_col="starttime", end_col="endtime"), "stay_id"])
res["delirium_drug"] = res.index.isin(rxs)
res["brain_pts"] = np.where(res["brain_gcs_pts"].notna() | res["delirium_drug"],
                            np.fmax(res["brain_gcs_pts"].fillna(0), res["delirium_drug"].astype(int)), np.nan)

# ============================ respiratory ============================
spec = lab[lab["itemid"] == 52033][["stay_id", "specimen_id", "value"]].rename(columns={"value": "spec_type"})
spec = spec.drop_duplicates(["stay_id", "specimen_id"])
pa = lab[lab["itemid"] == 50821][["stay_id", "specimen_id", "charttime", "valuenum"]].rename(columns={"valuenum": "pao2"})
pa["pao2"] = rng(pa["pao2"], "pao2")
pa = pa.merge(spec, on=["stay_id", "specimen_id"], how="left")
pa = pa[pa["spec_type"].fillna("").str.strip() == "ART."]
fl = lab[lab["itemid"] == 50816][["stay_id", "specimen_id", "valuenum"]].rename(columns={"valuenum": "fio2_lab"})
fl["fio2_lab"] = fio2_percent(fl["fio2_lab"])
pa = pa.merge(fl.drop_duplicates(["stay_id", "specimen_id"]), on=["stay_id", "specimen_id"], how="left")
fc = chart[chart["itemid"] == 223835][["stay_id", "charttime", "valuenum"]].rename(columns={"valuenum": "fio2_chart"})
fc["fio2_chart"] = fio2_percent(fc["fio2_chart"])
fc = fc[fc["fio2_chart"].notna()].sort_values("charttime")
pa = pd.merge_asof(pa.sort_values("charttime"), fc, on="charttime", by="stay_id", direction="backward", tolerance=4 * H)
pa["fio2"] = pa["fio2_lab"].fillna(pa["fio2_chart"])
pa = pa[pa["pao2"].notna() & pa["fio2"].notna()].copy()
pa["ratio"] = pa["pao2"] / (pa["fio2"] / 100.0)
pa["kind"] = "pf"

sp = chart[chart["itemid"] == 220277][["stay_id", "charttime", "valuenum"]].rename(columns={"valuenum": "spo2"})
sp["spo2"] = rng(sp["spo2"], "spo2")
sp = sp[sp["spo2"].notna() & (sp["spo2"] < 98) & cin(sp)]                      # fallback only if SpO2 < 98 %
sp = pd.merge_asof(sp.sort_values("charttime"), fc, on="charttime", by="stay_id", direction="backward", tolerance=4 * H)
sp = sp[sp["fio2_chart"].notna()].copy()
pft = pa[["stay_id", "charttime"]].rename(columns={"charttime": "pf_t"}).sort_values("pf_t")
if len(sp) and len(pft):                                                        # only if no arterial P/F within +-60 min
    sp = pd.merge_asof(sp.sort_values("charttime"), pft, left_on="charttime", right_on="pf_t",
                       by="stay_id", direction="nearest", tolerance=60 * MIN)
    sp = sp[sp["pf_t"].isna()]
sp["ratio"] = sp["spo2"] / (sp["fio2_chart"] / 100.0)
sp["kind"] = "sf"
meas = pd.concat([pa[["stay_id", "charttime", "ratio", "kind"]], sp[["stay_id", "charttime", "ratio", "kind"]]], ignore_index=True)
meas = meas[cin(meas)].reset_index(drop=True)

# advanced ventilatory support at each measurement time
inv = proc[proc["itemid"] == 225792]
niv = proc[proc["itemid"] == 225794]
dev = chart[chart["itemid"] == 226732][["stay_id", "charttime", "value"]].rename(columns={"charttime": "t"})
dev["dev"] = dev["value"].str.strip().str.lower()
nivc = chart[((chart["itemid"] == 227577) & (chart["value"].str.strip() != "Not applicable"))
             | ((chart["itemid"] == 227583) & (chart["value"].str.strip() == "On"))
             | ((chart["itemid"] == 225949) & (chart["value"].str.strip() != "Not applicable"))][["stay_id", "charttime"]].rename(columns={"charttime": "t"})
dv = asof_value(meas, dev, "dev", 6 * H)
meas["support"] = (covered(meas, inv) | covered(meas, niv)
                   | pd.Series(dv).isin(["high flow nasal cannula", "bipap mask", "cpap mask"]).to_numpy()
                   | asof_has(meas, nivc, 4 * H))
sup = meas["support"]; x = meas["ratio"]
pf_pts = np.select([(x <= 75) & sup, (x <= 150) & sup, x <= 225, x <= 300], [4, 3, 2, 1], default=0)   # no support: capped at 2
sf_pts = np.select([(x <= 120) & sup, (x <= 200) & sup, x <= 250, x <= 300], [4, 3, 2, 1], default=0)
meas["pts"] = np.where(meas["kind"] == "pf", pf_pts, sf_pts)
res["resp_pf_pts"] = meas[meas["kind"] == "pf"].groupby("stay_id")["pts"].max()
res["resp_sf_pts"] = meas[meas["kind"] == "sf"].groupby("stay_id")["pts"].max()
res["resp_meas_pts"] = meas.groupby("stay_id")["pts"].max()

# ECMO (respiratory 4; VA/VAV also cardiovascular 4)
ce = chart[cin(chart)]
ec_rows = ce[((ce["itemid"].isin([229270, 224660])) & (ce["valuenum"] > 0))
             | ((ce["itemid"] == 228193) & (ce["valuenum"] >= 1)) | (ce["itemid"] == 229268)]
ecs = set(ec_rows["stay_id"])
ecl = proc[proc["itemid"].isin([229529, 229530])]
ecs |= set(ecl.loc[in_window(ecl, coh, mode, start_col="starttime", end_col="endtime"), "stay_id"])
cfg = ce[(ce["itemid"] == 229268) & ce["value"].str.strip().str.upper().isin(["VA", "VAV"])]
vas = set(cfg["stay_id"])
res["ecmo"] = res.index.isin(ecs); res["ecmo_va"] = res.index.isin(vas)
res["resp_pts"] = np.where(res["ecmo"], 4, res["resp_meas_pts"])

# ============================ cardiovascular ============================
mp = chart[chart["itemid"].isin([220052, 220181, 225312])].copy()
mp["v"] = rng(mp["valuenum"], "map")
mp = mp[mp["v"].notna() & cin(mp)]
res["map_min"] = mp.groupby("stay_id")["v"].min()
DR = {221906: "ne", 221289: "epi", 221662: "dopa", 221653: "dobu", 222315: "vaso", 221749: "phen",
      229630: "phen", 229632: "phen", 221986: "mil", 229709: "ang", 229764: "ang"}
iv = inp[inp["itemid"].isin(list(DR)) & (inp["ordercategorydescription"] == "Continuous Med")
         & (inp["statusdescription"] != "Paused") & (inp["rate"] > 0)].copy()
iv["drug"] = iv["itemid"].map(DR)
iv["vr"] = np.nan
for dn, key in [("ne", "norepinephrine"), ("epi", "epinephrine"), ("dopa", "dopamine")]:
    m_ = (iv["drug"] == dn) & (iv["rateuom"] == "mcg/kg/min")
    iv.loc[m_, "vr"] = rng(iv.loc[m_, "rate"], key)
iv.loc[iv["drug"] == "ne", "vr"] = iv.loc[iv["drug"] == "ne", "vr"] * NE_FACTOR
if len(iv):
    iv = add_union(iv, ["stay_id", "drug"])
    iv = iv[iv["dur_h"] >= 1].copy()                                            # continuous infusion >= 1 h
    wlo, whi = iv["stay_id"].map(WLO), iv["stay_id"].map(WHI)
    iv["s"] = iv["starttime"].where(iv["starttime"] > wlo, wlo)
    iv["e"] = iv["endtime"].where(iv["endtime"] < whi, whi)
    iv = iv[iv["s"] < iv["e"]]
if len(iv):
    cand = iv[["stay_id", "s"]].drop_duplicates().rename(columns={"s": "t"})
    mm = cand.merge(iv[["stay_id", "drug", "vr", "s", "e"]], on="stay_id")
    mm = mm[(mm["s"] <= mm["t"]) & (mm["t"] < mm["e"])].copy()
    mm["is_ne"] = mm["drug"].isin(["ne", "epi"])
    mm["N"] = np.where(mm["is_ne"], mm["vr"].fillna(0), 0.0)
    mm["f_oth"] = mm["drug"].isin(["dobu", "vaso", "phen", "mil", "ang"])
    mm["f_dopa"] = mm["drug"] == "dopa"
    mm["dopa_r"] = np.where(mm["f_dopa"], mm["vr"], np.nan)
    ag = mm.groupby(["stay_id", "t"]).agg(N=("N", "sum"), has_ne=("is_ne", "max"), has_oth=("f_oth", "max"),
                                          has_dopa=("f_dopa", "max"), dopa_r=("dopa_r", "max")).reset_index()
    other = ag["has_oth"] | ag["has_dopa"]
    high = ag["N"] > 0.4; med = (ag["N"] > 0.2) & (ag["N"] <= 0.4)
    pts_ne = np.select([high | (med & other), med | other], [4, 3], default=2)
    dopa_only = (~ag["has_ne"]) & (~ag["has_oth"]) & ag["has_dopa"]
    pts_dopa = np.select([ag["dopa_r"] > 40, ag["dopa_r"] > 20], [4, 3], default=2)
    ag["pts"] = np.where(ag["has_ne"], pts_ne, np.where(ag["has_oth"], 2, np.where(dopa_only, pts_dopa, 2)))
    res["cv_infusion_pts"] = ag.groupby("stay_id")["pts"].max()
else:
    res["cv_infusion_pts"] = np.nan
cm = ce[((ce["itemid"] == 227980) & ~ce["value"].str.strip().str.lower().isin(["off", ""]))
        | (ce["itemid"].isin([228154, 229671, 220125, 220128, 229254]) & (ce["valuenum"] > 0))]
mcs = set(cm["stay_id"]) | vas
res["mech_support"] = res.index.isin(mcs)
map_pts = np.where(res["map_min"] < 70, 1, 0)
cv = np.where(res["cv_infusion_pts"].notna(), np.fmax(res["cv_infusion_pts"].fillna(0), map_pts),
              np.where(res["map_min"].notna(), map_pts, np.nan))
res["cv_pts"] = np.where(res["mech_support"], 4, cv)

# ============================ kidney ============================
wl = []
for itm, prio, f in [(226512, 1, 1.0), (224639, 2, 1.0), (226531, 3, 1 / 2.2046)]:
    d_ = chart[chart["itemid"] == itm][["stay_id", "charttime", "valuenum"]].copy()
    d_["kg"] = d_["valuenum"] * f; d_["prio"] = prio; d_["t"] = d_["charttime"]
    wl.append(d_[["stay_id", "t", "kg", "prio"]])
iw = inp[["stay_id", "starttime", "patientweight"]].copy()
iw["kg"] = pd.to_numeric(iw["patientweight"], errors="coerce"); iw["prio"] = 4; iw["t"] = iw["starttime"]
wl.append(iw[["stay_id", "t", "kg", "prio"]])
W = pd.concat(wl, ignore_index=True)
W["kg"] = rng(W["kg"], "weight"); W = W[W["kg"].notna()]
weight = W.sort_values(["stay_id", "prio", "t"]).drop_duplicates("stay_id").set_index("stay_id")["kg"]
res["weight_kg"] = weight

uo_items = [226559, 226560, 226561, 226563, 226564, 226565, 226567, 226584, 226557, 226558, 227489, 227488]
o = outp[outp["itemid"].isin(uo_items)].copy()
o["v"] = rng(o["value"], "uo_record")
o.loc[o["itemid"] == 227488, "v"] = -o.loc[o["itemid"] == 227488, "v"]
o = o[o["v"].notna() & cin(o)]
o["hr"] = np.floor((o["charttime"] - o["stay_id"].map(WLO)) / H).astype(int)
bins = o.groupby(["stay_id", "hr"])["v"].sum().clip(lower=0).reset_index().sort_values(["stay_id", "hr"])
bins["kg"] = bins["stay_id"].map(weight)
bins["rate"] = bins["v"] / bins["kg"]

def longest_run(flag):
    f = bins[flag].copy()
    if f.empty:
        return pd.Series(dtype=float)
    f["k"] = f["hr"] - f.groupby("stay_id").cumcount()
    return f.groupby(["stay_id", "k"]).size().groupby("stay_id").max()
res["run_lt05"] = longest_run(bins["rate"] < 0.5)
res["run_lt03"] = longest_run(bins["rate"] < 0.3)
res["run_zero"] = longest_run(bins["v"] == 0)
for c_ in ["run_lt05", "run_lt03", "run_zero"]:
    res[c_] = res[c_].fillna(0)
res["uo_observed"] = res.index.isin(bins["stay_id"])

cr = res["creatinine_max"]
cr_pts = np.select([cr > 3.5, cr > 2.0, cr > 1.2], [3, 2, 1], default=np.where(cr.isna(), np.nan, 0))
uo_pts = np.select([res["run_lt03"] >= 24, res["run_lt05"] >= 12, res["run_lt05"] >= 6], [3, 2, 1], default=0)
base_pts = np.fmax(np.nan_to_num(cr_pts, nan=0), uo_pts)
rrt_p = proc[proc["itemid"].isin([225441, 225802, 225805])]
rrt = set(rrt_p.loc[in_window(rrt_p, coh, mode, start_col="starttime", end_col="endtime"), "stay_id"])
rrt |= set(ce.loc[ce["itemid"] == 227290, "stay_id"])
chronic = set(dxd["stay_id"]) | set(ce.loc[(ce["itemid"] == 225126) & (ce["valuenum"] == 1), "stay_id"])
res["rrt_receiving"] = res.index.isin(rrt); res["chronic_dialysis"] = res.index.isin(chronic)
crit_a = (res["creatinine_max"] > 1.2) | (res["run_lt03"] > 6)
crit_b = (res["potassium_max"] >= 6.0) | ((res["ph_min"] <= 7.20) & (res["bicarbonate_min"] <= 12))
res["rrt_criteria_only"] = (crit_a & crit_b) & ~res["rrt_receiving"] & ~res["chronic_dialysis"]
four_core = res["rrt_receiving"] | res["chronic_dialysis"] | (res["run_zero"] >= 12)
res["renal_pts"] = np.where(four_core | res["rrt_criteria_only"], 4, base_pts)
res["renal_pts_norrt_criteria"] = np.where(four_core, 4, base_pts)
obs_k = res["creatinine_max"].notna() | res["uo_observed"] | res["rrt_receiving"] | res["chronic_dialysis"]
res.loc[~obs_k, ["renal_pts", "renal_pts_norrt_criteria"]] = np.nan

# ============================ totals ============================
dom = ["brain_pts", "resp_pts", "cv_pts", "liver_pts", "renal_pts", "hemo_pts"]
res["n_domains_observed"] = res[dom].notna().sum(axis=1)
res["complete_case"] = res["n_domains_observed"] == 6
res["sofa2_total"] = res[dom].fillna(0).sum(axis=1).astype(int)
alt = [c if c != "renal_pts" else "renal_pts_norrt_criteria" for c in dom]
res["sofa2_total_norrt_criteria"] = res[alt].fillna(0).sum(axis=1).astype(int)
res.reset_index().to_csv(f"outputs/tables/sofa2_{tag}_PRIVATE.csv", index=False)

# ============================ aggregate summary (safe to share) ============================
n = len(res)
print(f"SOFA-2 ({tag})  N stays = {n:,}")
rows = []
for dname in dom:
    vc = res[dname].value_counts().reindex([0, 1, 2, 3, 4], fill_value=0)
    obs = int(res[dname].notna().sum())
    rows.append({"domain": dname, "n_observed": obs, "pct_missing": round(100 * (1 - obs / n), 1),
                 **{f"pts_{k}": int(v) for k, v in vc.items()}})
summ = pd.DataFrame(rows); print(summ.to_string(index=False))
t = res["sofa2_total"]
print(f"\nTotal SOFA-2 (missing = 0): median {t.median():.0f} [IQR {t.quantile(.25):.0f}, {t.quantile(.75):.0f}], mean {t.mean():.2f}, max {t.max()}")
print(f"complete-case stays: {int(res['complete_case'].sum()):,}")
tc = t[res["complete_case"]]
if len(tc):
    print(f"complete-case total: median {tc.median():.0f} [IQR {tc.quantile(.25):.0f}, {tc.quantile(.75):.0f}]")
print("\nSOFA-2-specific features (stays):")
print(f"  delirium-drug point raised brain score: {int(((res['delirium_drug']) & (res['brain_gcs_pts'].fillna(0) < 1)).sum()):,}")
print(f"  SpO2/FiO2 fallback available (no P/F score): {int((res['resp_sf_pts'].notna() & res['resp_pf_pts'].isna()).sum()):,}")
print(f"  ECMO: {int(res['ecmo'].sum()):,} (VA/VAV {int(res['ecmo_va'].sum()):,}); mechanical circulatory support: {int(res['mech_support'].sum()):,}")
print(f"  RRT receiving: {int(res['rrt_receiving'].sum()):,}; chronic dialysis: {int(res['chronic_dialysis'].sum()):,}; "
      f"4 points from RRT criteria only: {int(res['rrt_criteria_only'].sum()):,}")
print(f"  weight available for per-kg urine output: {int(res['weight_kg'].notna().sum()):,} stays")

s1f = f"outputs/tables/sofa1_{mode}_PRIVATE.csv"
try:
    s1 = pd.read_csv(s1f).set_index("stay_id")
    dlt = res["sofa2_total"] - s1["sofa1_total"].reindex(res.index)
    print(f"\nSOFA-2 minus SOFA-1 (same stays, {mode} window): equal {100*(dlt==0).mean():.1f}%, "
          f"SOFA-2 higher {100*(dlt>0).mean():.1f}% (median +{dlt[dlt>0].median():.0f}), "
          f"SOFA-2 lower {100*(dlt<0).mean():.1f}% (median {dlt[dlt<0].median():.0f})")
    print("paper benchmark (eICU): equal 49%, higher 11% (median +2), lower 40% (median -3)")
    mp_ = {"brain_pts": "cns_pts", "resp_pts": "resp_pts", "cv_pts": "cv_pts", "liver_pts": "liver_pts",
           "renal_pts": "renal_pts", "hemo_pts": "coag_pts"}
    print("mean points per domain (missing=0):  SOFA-1 -> SOFA-2")
    for d2, d1 in mp_.items():
        print(f"  {d2:10s} {s1[d1].fillna(0).mean():.3f} -> {res[d2].fillna(0).mean():.3f}")
except FileNotFoundError:
    print("\n(SOFA-1 file not found; skipped comparison)")
summ.to_csv(f"outputs/tables/sofa2_{tag}_domain_summary.csv", index=False)
t.value_counts().sort_index().rename_axis("sofa2_total").rename("n").reset_index().to_csv(f"outputs/tables/sofa2_{tag}_distribution.csv", index=False)
print(f"\nSaved sofa2_{tag}_PRIVATE.csv (stay-level, keep local) and aggregate summaries in outputs/tables/")
