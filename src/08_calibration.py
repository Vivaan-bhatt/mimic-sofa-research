# 08_calibration.py -- calibration analysis of SOFA-1 and SOFA-2 for mortality, overall and by subgroup.
# SOFA is an ordinal organ-dysfunction score, NOT a fitted probability model. We therefore report:
#   (1) observed mortality by score band (descriptive, Wilson CIs);
#   (2) for each score, a descriptive logistic calibration curve P(death) = expit(a + b*score) fitted on ALL stays;
#   (3) subgroup calibration-in-the-large: O/E = observed deaths / deaths expected from that overall curve;
#   (4) subgroup calibration slope (logit change per point) and its ratio to the reference group's slope;
#   (5) CALIBRATION DISPARITY = log(O/E_group) - log(O/E_reference) and its change from SOFA-1 to SOFA-2,
#       with paired patient-cluster bootstrap CIs (same replicates for both scores).
# No model is tuned to improve how calibration looks. Run only after the protocol is frozen.
#   python3 src/08_calibration.py --outcome hospital_death --window strict
#   python3 src/08_calibration.py --selftest
import argparse, importlib.util, json, os, subprocess, sys, time
import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("auc06", os.path.join(here, "06_auc_bootstrap.py"))
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
NB, SCORES, COMPS, HEADLINE = A.NB, A.SCORES, A.COMPS, A.HEADLINE
ORDER, PRIMARY_LEVELS, REF = A.ORDER, A.PRIMARY_LEVELS, A.REF
X = np.arange(NB, dtype=float)
BANDS = [(0, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11),
         (12, 12), (13, 14), (15, 16), (17, 19), (20, 24)]          # fixed in advance, outcome-blind

def logistic_fit(pos, neg, iters=40):
    """Newton-Raphson for P(death)=expit(a+b*x) on grouped data; pos/neg shape (L, 25). Returns a, b (L,) with NaN if degenerate."""
    pos, neg = np.atleast_2d(pos).astype(float), np.atleast_2d(neg).astype(float)
    n = pos + neg
    P, N = pos.sum(1), neg.sum(1)
    ok = (P > 0) & (N > 0) & ((n > 0).sum(1) >= 2)
    a = np.where(ok, np.log(np.where(ok, P / np.where(N > 0, N, 1), 1)), np.nan); b = np.zeros(len(P))
    with np.errstate(all="ignore"):
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(a[:, None] + b[:, None] * X)))
            r = pos - n * p
            ga, gb = r.sum(1), (X * r).sum(1)
            w = n * p * (1 - p)
            haa, hab, hbb = -w.sum(1), -(w * X).sum(1), -(w * X * X).sum(1)
            det = haa * hbb - hab ** 2
            da, db = -(hbb * ga - hab * gb) / det, -(-hab * ga + haa * gb) / det
            da, db = np.clip(da, -5, 5), np.clip(db, -2, 2)
            a, b = a + np.nan_to_num(da), b + np.nan_to_num(db)
    a[~ok], b[~ok] = np.nan, np.nan
    return a, b

def hist_all(df, keys, w, y):
    posw, negw = w * y, w * (1 - y)
    out = {}
    for p in ORDER:
        L = len(ORDER[p])
        for s in SCORES:
            pos = np.bincount(keys[p][s], weights=posw, minlength=L * NB).reshape(L, NB)
            neg = np.bincount(keys[p][s], weights=negw, minlength=L * NB).reshape(L, NB)
            out[(p, s)] = (pos, neg)
    return out

def stats(H):
    """From grouped histograms -> {(partition, score): dict(oe, slope)} per level; overall curve from the sex partition."""
    res = {}
    for s in SCORES:
        pos_all, neg_all = H[("sex", s)][0].sum(0, keepdims=True), H[("sex", s)][1].sum(0, keepdims=True)
        a0, b0 = logistic_fit(pos_all, neg_all)
        pcurve = 1 / (1 + np.exp(-(a0[0] + b0[0] * X)))
        res[("overall", s)] = dict(slope=b0, intercept=a0)
        for p in ORDER:
            pos, neg = H[(p, s)]
            n = pos + neg
            E = (n * pcurve).sum(1); O = pos.sum(1)
            a, b = logistic_fit(pos, neg)
            with np.errstate(all="ignore"):
                res[(p, s)] = dict(oe=O / E, slope=b, slope_ratio=b / b0[0])
    return res

def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

def run(df, B, seed, outcome):
    y = df["y"].to_numpy().astype(float)
    subj_idx, subj_uni = pd.factorize(df["subject_id"]); S = len(subj_uni)
    keys = {}
    for p in ORDER:
        codes = pd.Categorical(df[p], categories=ORDER[p]).codes.astype(np.int64)
        assert (codes >= 0).all()
        keys[p] = {s: codes * NB + df[s].to_numpy().astype(np.int64) for s in SCORES}
    point = stats(hist_all(df, keys, np.ones(len(df)), y))
    rng = np.random.default_rng(seed)
    boot = {k: {m: np.empty((B,) + np.shape(v)) for m, v in d.items()} for k, d in point.items()}
    t0 = time.time()
    for b in range(B):
        w = np.bincount(rng.integers(0, S, S), minlength=S).astype(float)[subj_idx]
        st = stats(hist_all(df, keys, w, y))
        for k, d in st.items():
            for m, v in d.items():
                boot[k][m][b] = v
        if (b + 1) % 250 == 0:
            print(f"  bootstrap {b+1}/{B} ({time.time()-t0:,.0f}s)", flush=True)
    ci = lambda a: (np.nanpercentile(a, 2.5, axis=0), np.nanpercentile(a, 97.5, axis=0))
    def pv(a):
        a = a[~np.isnan(a)]
        return min(1.0, 2 * (min((a <= 0).sum(), (a >= 0).sum()) + 1) / (len(a) + 1))
    rows = []
    for p in ORDER:
        for lvl in ORDER[p]:
            i = ORDER[p].index(lvl); sel = (df[p] == lvl).to_numpy()
            prim = lvl in PRIMARY_LEVELS[p]
            for s in SCORES:
                for m in ["oe", "slope"]:
                    pt, bt = point[(p, s)][m][i], boot[(p, s)][m][:, i]
                    lo, hi = ci(bt)
                    rows.append(dict(family=m, partition=p, level=lvl, ref="", score=s, comparator="", estimate=pt, lo=lo, hi=hi,
                                     p_boot=np.nan, n=int(sel.sum()), deaths=int(y[sel].sum()), in_primary_set=prim))
    for p in ORDER:
        ri = ORDER[p].index(REF[p])
        for lvl in PRIMARY_LEVELS[p]:
            if lvl == REF[p]:
                continue
            li = ORDER[p].index(lvl); sel = (df[p] == lvl).to_numpy()
            n, d = int(sel.sum()), int(y[sel].sum())
            disp_pt = {s: np.log(point[(p, s)]["oe"][li]) - np.log(point[(p, s)]["oe"][ri]) for s in SCORES}
            disp_bt = {s: np.log(boot[(p, s)]["oe"][:, li]) - np.log(boot[(p, s)]["oe"][:, ri]) for s in SCORES}
            rat_pt = {s: point[(p, s)]["slope_ratio"][li] / point[(p, s)]["slope_ratio"][ri] for s in SCORES}
            rat_bt = {s: boot[(p, s)]["slope_ratio"][:, li] / boot[(p, s)]["slope_ratio"][:, ri] for s in SCORES}
            for s in SCORES:
                lo, hi = ci(disp_bt[s])
                rows.append(dict(family="calib_disparity_logOE", partition=p, level=lvl, ref=REF[p], score=s, comparator="", estimate=disp_pt[s],
                                 lo=lo, hi=hi, p_boot=pv(disp_bt[s]), n=n, deaths=d, in_primary_set=True))
                lo, hi = ci(rat_bt[s])
                rows.append(dict(family="slope_ratio_vs_ref", partition=p, level=lvl, ref=REF[p], score=s, comparator="", estimate=rat_pt[s],
                                 lo=lo, hi=hi, p_boot=np.nan, n=n, deaths=d, in_primary_set=True))
            for c in COMPS:
                bt = disp_bt["sofa2"] - disp_bt[c]; pt = disp_pt["sofa2"] - disp_pt[c]; lo, hi = ci(bt)
                rows.append(dict(family="delta_calib_disparity_logOE", partition=p, level=lvl, ref=REF[p], score="sofa2 - comparator", comparator=c,
                                 estimate=pt, lo=lo, hi=hi, p_boot=pv(bt), n=n, deaths=d, in_primary_set=True))
                bt = rat_bt["sofa2"] - rat_bt[c]; pt = rat_pt["sofa2"] - rat_pt[c]; lo, hi = ci(bt)
                rows.append(dict(family="delta_slope_ratio", partition=p, level=lvl, ref=REF[p], score="sofa2 - comparator", comparator=c,
                                 estimate=pt, lo=lo, hi=hi, p_boot=pv(bt), n=n, deaths=d, in_primary_set=True))
    res = pd.DataFrame(rows); res["outcome"] = outcome
    res["tier"] = np.where(res["family"].str.startswith("delta") & (res["comparator"] == HEADLINE) & (outcome == "hospital_death"),
                           "primary", "co-primary/exploratory")
    # descriptive observed mortality by score band, overall and by subgroup
    brow = []
    cells = [("overall", "All stays", np.ones(len(df), bool))] + [(p, l, (df[p] == l).to_numpy()) for p in ORDER for l in ORDER[p]]
    for p, lvl, sel in cells:
        for s in SCORES:
            sc = df[s].to_numpy()
            for lo_, hi_ in BANDS:
                m_ = sel & (sc >= lo_) & (sc <= hi_)
                nn, kk = int(m_.sum()), int(y[m_].sum())
                l_, h_ = wilson(kk, nn)
                brow.append(dict(partition=p, level=lvl, score=s, band=f"{lo_}-{hi_}" if lo_ != hi_ else str(lo_), n=nn, deaths=kk,
                                 observed=kk / nn if nn else np.nan, wilson_lo=l_, wilson_hi=h_))
    return res, pd.DataFrame(brow), point

def fmt(r):
    return f"{r.estimate:.3f} ({r.lo:.3f} to {r.hi:.3f})"

def print_summary(res, point):
    print("\nOverall descriptive calibration curve  logit P(death) = a + b*score:")
    for s in SCORES:
        d = point[("overall", s)]; print(f"  {s:18s} a = {d['intercept'][0]:.3f}, b = {d['slope'][0]:.3f} per point")
    for c in COMPS:
        print(f"\nSubgroup O/E (observed / expected from the overall curve; 1 = matches pooled curve): sofa2 | {c}")
        t = res[(res.family == "oe") & res.in_primary_set & (res.partition != "overall")]
        for (p, l), g in t.groupby(["partition", "level"], sort=False):
            a = g[g.score == "sofa2"].iloc[0]; b = g[g.score == c].iloc[0]
            print(f"  {p:9s} {l:16s} n={a.n:>6,} | {fmt(a)} | {fmt(b)}")
        print(f"Change in calibration disparity, log(O/E group) - log(O/E reference), SOFA-2 minus {c}:")
        for r in res[(res.family == "delta_calib_disparity_logOE") & (res.comparator == c)].itertuples():
            print(f"  {r.partition:9s} {r.ref} vs {r.level:16s} {fmt(r)} p={r.p_boot:.3f}")

def selftest():
    print("SELF-TEST on synthetic data (no MIMIC data involved)")
    rng = np.random.default_rng(2); n = 30000
    subj = rng.integers(0, 22000, n)
    sex = rng.choice(["Male", "Female"], n, p=[.56, .44]); race = rng.choice(ORDER["race"], n, p=[.66, .11, .04, .03, .04, .12])
    ins = rng.choice(ORDER["insurance"], n, p=[.55, .15, .26, .04]); sev = rng.normal(0, 1, n)
    shift = np.where(race == "Black", 0.6, 0.0)                              # injected: higher mortality at the same severity
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.1 * sev - 2.3 + shift)))).astype(int)
    mk = lambda off, sd: np.clip(np.round(off + 2.2 * sev + rng.normal(0, sd, n)), 0, 24).astype(int)
    df = pd.DataFrame(dict(subject_id=subj, y=y, sex=sex, race=race, insurance=ins,
                           sofa1_community=mk(4, 1.6), sofa1_ett_charted=mk(5, 1.8), sofa2=mk(4.5, 1.5)))
    res, bands, point = run(df, B=300, seed=5, outcome="hospital_death")
    keys = {p: {s: pd.Categorical(df[p], categories=ORDER[p]).codes.astype(np.int64) * NB + df[s].to_numpy() for s in SCORES} for p in ORDER}
    H = hist_all(df, keys, np.ones(n), y.astype(float))
    for s in SCORES:
        pos, neg = H[("sex", s)][0].sum(0), H[("sex", s)][1].sum(0)
        a, b = logistic_fit(pos[None], neg[None]); nn = pos + neg; p_ = 1 / (1 + np.exp(-(a[0] + b[0] * X)))
        assert abs((pos - nn * p_).sum()) < 1e-6 and abs((X * (pos - nn * p_)).sum()) < 1e-6, "score equations not satisfied"
        ll = lambda aa, bb: (pos * np.log(1 / (1 + np.exp(-(aa + bb * X)))) + neg * np.log(1 - 1 / (1 + np.exp(-(aa + bb * X))))).sum()
        assert all(ll(a[0], b[0]) >= ll(a[0] + da, b[0] + db) - 1e-9 for da in (-.02, 0, .02) for db in (-.003, 0, .003)), "not a maximum"
        tot = sum(res[(res.family == "oe") & (res.partition == "sex") & (res.score == s)].estimate.to_numpy() * 0 + 1)
        o = res[(res.family == "oe") & (res.partition == "sex") & (res.score == s)]
        ratio_all = (o.estimate * (o.n)).sum()   # not used; overall O/E checked below via deaths
        d_all = o.deaths.sum(); e_all = (o.deaths / o.estimate).sum()
        assert abs(d_all / e_all - 1) < 1e-6, "overall O/E must equal 1 for the pooled MLE curve"
    g = res[(res.family == "oe") & (res.partition == "race") & (res.level == "Black") & (res.score == "sofa2")].iloc[0]
    assert g.estimate > 1.15 and g.lo > 1.0, g
    r2, _, _ = run(df.assign(sofa1_community=df["sofa2"]), B=60, seed=3, outcome="hospital_death")
    z = r2[(r2.family == "delta_calib_disparity_logOE") & (r2.comparator == "sofa1_community")]
    assert (z.estimate.abs() < 1e-12).all() and (z.lo.abs() < 1e-12).all() and (z.hi.abs() < 1e-12).all()
    tot_n = bands[(bands.partition == "overall") & (bands.score == "sofa2")].n.sum()
    assert tot_n == n, "bands must cover every stay exactly once"
    print("all checks passed: score equations satisfied and curve is a local maximum; pooled O/E = 1; injected excess mortality "
          f"detected (Black O/E {g.estimate:.2f}); identical scores give zero change; score bands partition all stays")
    print_summary(res, point)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="hospital_death", choices=["hospital_death", "icu_death_strict"])
    ap.add_argument("--window", default="strict"); ap.add_argument("--sofa2-tag", default=None)
    ap.add_argument("--sofa2-col", default="sofa2_total"); ap.add_argument("--complete-case", action="store_true")
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--seed", type=int, default=20251007)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    t2 = a.sofa2_tag or a.window
    coh = pd.read_csv("outputs/tables/cohort_PRIVATE_do_not_share.csv").set_index("stay_id")
    f = lambda name: pd.read_csv(f"outputs/tables/{name}_PRIVATE.csv").set_index("stay_id")
    s1c, s1e, s2 = f(f"sofa1_{a.window}"), f(f"sofa1_{a.window}_ett_charted"), f(f"sofa2_{t2}")
    df = pd.DataFrame({"subject_id": coh["subject_id"], "y": coh[a.outcome].astype(int),
                       "sex": coh["gender"].map({"M": "Male", "F": "Female"}), "race": coh["race"].map(A.race_group),
                       "insurance": coh["insurance"].fillna("Other/Missing").where(coh["insurance"].isin(["Medicare", "Medicaid", "Private"]), "Other/Missing"),
                       "sofa1_community": s1c["sofa1_total"], "sofa1_ett_charted": s1e["sofa1_total"], "sofa2": s2[a.sofa2_col]})
    assert df.notna().all().all()
    if a.complete_case:
        df = df[(s1c["complete_case"] & s1e["complete_case"] & s2["complete_case"]).reindex(df.index).fillna(False)]
    print(f"N stays = {len(df):,}; deaths ({a.outcome}) = {int(df.y.sum()):,}; window = {a.window}; sofa2 = {t2}/{a.sofa2_col}; B = {a.B}")
    res, bands, point = run(df.reset_index(drop=True), a.B, a.seed, a.outcome)
    tag = f"{a.outcome}_{a.window}" + (f"_{t2}" if a.sofa2_tag else "") + ("_cc" if a.complete_case else "")
    res.to_csv(f"outputs/tables/calib_results_{tag}.csv", index=False); bands.to_csv(f"outputs/tables/calib_bands_{tag}.csv", index=False)
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git = "not a git repository"
    json.dump({"outcome": a.outcome, "window": a.window, "sofa2_tag": t2, "B": a.B, "seed": a.seed, "n_stays": len(df), "git_commit": git,
               "method": "descriptive logistic calibration curve per score; O/E by subgroup; paired subject-cluster bootstrap"},
              open(f"outputs/tables/calibration_manifest_{tag}.json", "w"), indent=2)
    print_summary(res, point)
    print(f"\nSaved calib_results_{tag}.csv, calib_bands_{tag}.csv and calibration_manifest_{tag}.json in outputs/tables/")

if __name__ == "__main__":
    main()
