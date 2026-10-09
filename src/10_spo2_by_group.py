# 10_spo2_by_group.py -- does the SOFA-2 SpO2/FiO2 fallback score patients differently from the arterial PaO2/FiO2 score,
# and does that differ by sex, race or insurance?  Uses PAIRED measurements: an arterial P/F and an SpO2 reading (< 98 %)
# within 10 minutes, with the SAME FiO2, so the only difference is the oxygenation measure. Exploratory, outcome-blind.
#   python3 src/10_spo2_by_group.py          python3 src/10_spo2_by_group.py --selftest
import argparse, importlib.util, os, sys
import numpy as np
import pandas as pd
from common import *

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("auc06", os.path.join(here, "06_auc_bootstrap.py"))
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)

def pts_pf(x, sup):
    return np.select([(x <= 75) & sup, (x <= 150) & sup, x <= 225, x <= 300], [4, 3, 2, 1], default=0)
def pts_sf(x, sup):
    return np.select([(x <= 120) & sup, (x <= 200) & sup, x <= 250, x <= 300], [4, 3, 2, 1], default=0)

def boot_mean(d, w_groups, subj_idx, S, B, seed):
    """Cluster bootstrap of the stay-level mean difference within each group (rows of w_groups are group masks)."""
    rng = np.random.default_rng(seed); out = np.empty((B, len(w_groups)))
    for b in range(B):
        w = np.bincount(rng.integers(0, S, S), minlength=S).astype(float)[subj_idx]
        for j, m in enumerate(w_groups):
            ww = w * m
            out[b, j] = (ww * d).sum() / ww.sum() if ww.sum() > 0 else np.nan
    return out

def selftest():
    print("SELF-TEST (no MIMIC data involved)")
    sup, no = np.array([True]), np.array([False])
    assert pts_pf(np.array([75]), sup)[0] == 4 and pts_pf(np.array([75]), no)[0] == 2        # no support: capped at 2
    assert pts_pf(np.array([150]), sup)[0] == 3 and pts_pf(np.array([225]), no)[0] == 2
    assert pts_pf(np.array([300]), no)[0] == 1 and pts_pf(np.array([301]), no)[0] == 0
    assert pts_sf(np.array([120]), sup)[0] == 4 and pts_sf(np.array([121]), sup)[0] == 3
    assert pts_sf(np.array([200]), sup)[0] == 3 and pts_sf(np.array([250]), no)[0] == 2 and pts_sf(np.array([300]), no)[0] == 1
    x = np.arange(20, 500, 5.0)
    for fn in (pts_pf, pts_sf):
        for s in (True, False):
            p = fn(x, np.full(len(x), s)); assert (np.diff(p) <= 0).all(), "points must not rise as the ratio rises"
    rng = np.random.default_rng(0); n = 4000
    d = rng.normal(0, 1, n) + np.where(np.arange(n) % 2 == 0, -0.3, 0.0)                 # group 0 under-scored by 0.3 points
    g0 = (np.arange(n) % 2 == 0).astype(float); subj = np.arange(n)
    bt = boot_mean(d, [g0, 1 - g0], subj, n, 300, 1)
    lo, hi = np.nanpercentile(bt[:, 0], [2.5, 97.5])
    assert lo < -0.3 < hi + 0.05 and hi < 0, (lo, hi)
    print("all checks passed: point thresholds and support cap match Table 2; points never rise with the ratio; bootstrap recovers an injected -0.3 point under-scoring")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--window", default="strict"); ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20251007); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    mode = a.window
    coh = load_cohort(); cov = coh.set_index("stay_id")
    chart = read_extract("chart_long", time_cols=["charttime"], num_cols=["valuenum"])
    lab = read_extract("lab_long", time_cols=["charttime"], num_cols=["valuenum"])
    proc = read_extract("proc_long", time_cols=["starttime", "endtime"])
    # arterial P/F rows, built exactly as in 04_sofa2.py
    spec = lab[lab["itemid"] == 52033][["stay_id", "specimen_id", "value"]].rename(columns={"value": "spec_type"}).drop_duplicates(["stay_id", "specimen_id"])
    pa = lab[lab["itemid"] == 50821][["stay_id", "specimen_id", "charttime", "valuenum"]].rename(columns={"valuenum": "pao2"})
    pa["pao2"] = rng(pa["pao2"], "pao2")
    pa = pa.merge(spec, on=["stay_id", "specimen_id"], how="left")
    pa = pa[pa["spec_type"].fillna("").str.strip() == "ART."]
    fl = lab[lab["itemid"] == 50816][["stay_id", "specimen_id", "valuenum"]].rename(columns={"valuenum": "fio2_lab"})
    fl["fio2_lab"] = fio2_percent(fl["fio2_lab"])
    pa = pa.merge(fl.drop_duplicates(["stay_id", "specimen_id"]), on=["stay_id", "specimen_id"], how="left")
    fc = chart[chart["itemid"] == 223835][["stay_id", "charttime", "valuenum"]].rename(columns={"valuenum": "fio2_chart"})
    fc["fio2_chart"] = fio2_percent(fc["fio2_chart"]); fc = fc[fc["fio2_chart"].notna()].sort_values("charttime")
    pa = pd.merge_asof(pa.sort_values("charttime"), fc, on="charttime", by="stay_id", direction="backward", tolerance=4 * H)
    pa["fio2"] = pa["fio2_lab"].fillna(pa["fio2_chart"])
    pa = pa[pa["pao2"].notna() & pa["fio2"].notna() & in_window(pa, coh, mode, time_col="charttime")].copy()
    pa["pf"] = pa["pao2"] / (pa["fio2"] / 100.0)
    # SpO2 readings < 98 % paired within 10 minutes, same FiO2
    sp = chart[chart["itemid"] == 220277][["stay_id", "charttime", "valuenum"]].rename(columns={"charttime": "sp_t", "valuenum": "spo2"})
    sp["spo2"] = rng(sp["spo2"], "spo2"); sp = sp[sp["spo2"].notna() & (sp["spo2"] < 98)].sort_values("sp_t")
    pr = pd.merge_asof(pa.sort_values("charttime"), sp, left_on="charttime", right_on="sp_t", by="stay_id", direction="nearest", tolerance=10 * MIN)
    pr = pr[pr["spo2"].notna()].copy().reset_index(drop=True)
    pr["sf"] = pr["spo2"] / (pr["fio2"] / 100.0)
    inv, niv = proc[proc["itemid"] == 225792], proc[proc["itemid"] == 225794]
    dev = chart[chart["itemid"] == 226732][["stay_id", "charttime", "value"]].rename(columns={"charttime": "t"}); dev["dev"] = dev["value"].str.strip().str.lower()
    nivc = chart[((chart["itemid"] == 227577) & (chart["value"].str.strip() != "Not applicable")) | ((chart["itemid"] == 227583) & (chart["value"].str.strip() == "On"))
                 | ((chart["itemid"] == 225949) & (chart["value"].str.strip() != "Not applicable"))][["stay_id", "charttime"]].rename(columns={"charttime": "t"})
    dv = asof_value(pr, dev, "dev", 6 * H)
    sup = (covered(pr, inv) | covered(pr, niv) | pd.Series(dv).isin(["high flow nasal cannula", "bipap mask", "cpap mask"]).to_numpy() | asof_has(pr, nivc, 4 * H))
    pr["pts_pf"], pr["pts_sf"] = pts_pf(pr["pf"].to_numpy(), sup), pts_sf(pr["sf"].to_numpy(), sup)
    pr["diff"] = pr["pts_sf"] - pr["pts_pf"]
    print(f"paired measurements: {len(pr):,} from {pr['stay_id'].nunique():,} stays; mean S/F-points minus P/F-points = {pr['diff'].mean():+.3f}; "
          f"S/F scores lower in {100*(pr['diff']<0).mean():.1f}%, higher in {100*(pr['diff']>0).mean():.1f}%, equal in {100*(pr['diff']==0).mean():.1f}%")
    st = pr.groupby("stay_id")["diff"].mean()
    idx = st.index
    subj_idx, subj_uni = pd.factorize(cov.loc[idx, "subject_id"]); S = len(subj_uni)
    ins = cov["insurance"].fillna("Other/Missing"); ins = ins.where(ins.isin(["Medicare", "Medicaid", "Private"]), "Other/Missing")
    groups = {"sex": cov["gender"].map({"M": "Male", "F": "Female"}), "race": cov["race"].map(A.race_group), "insurance": ins}
    s2 = pd.read_csv(f"outputs/tables/sofa2_{mode}_sed_broad_PRIVATE.csv").set_index("stay_id")
    rows = []
    for p, g in groups.items():
        for lvl in A.ORDER[p]:
            gm = (g.reindex(idx) == lvl).to_numpy().astype(float)
            n_st = int(gm.sum())
            allst = g.index[g == lvl]
            sfonly = int((s2.loc[allst, "resp_sf_pts"].notna() & s2.loc[allst, "resp_pf_pts"].isna()).sum())
            r = dict(partition=p, level=lvl, n_stays_in_group=len(allst), pct_resp_from_spo2_only=round(100 * sfonly / max(1, len(allst)), 1), n_stays_paired=n_st)
            if n_st >= 10:
                pm = pr[pr["stay_id"].isin(idx[gm.astype(bool)])]
                bt = boot_mean(st.to_numpy(), [gm], subj_idx, S, a.B, a.seed)[:, 0]
                lo, hi = np.nanpercentile(bt, [2.5, 97.5])
                r.update(n_pairs=len(pm), mean_diff=float(st[gm.astype(bool)].mean()), lo=lo, hi=hi,
                         pct_sf_lower=round(100 * (pm["diff"] < 0).mean(), 1), pct_sf_higher=round(100 * (pm["diff"] > 0).mean(), 1))
            rows.append(r)
    out = pd.DataFrame(rows); out.to_csv("outputs/tables/spo2_vs_pao2_by_group.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print("\nPaired SpO2/FiO2- vs PaO2/FiO2-derived SOFA-2 respiratory points (negative = SpO2 route scores LOWER); % of stays whose respiratory score comes from SpO2 only")
    print(out.drop(columns=[]).round(3).to_string(index=False))
    print("\nSaved outputs/tables/spo2_vs_pao2_by_group.csv (aggregates only)")

if __name__ == "__main__":
    main()
