# 06_auc_bootstrap.py -- AUROC, paired DeltaAUROC, subgroup gaps and change in gaps, with a paired
# patient-cluster bootstrap. SOFA-1 and SOFA-2 are always evaluated on the same stays, the same outcome and the
# same window; each bootstrap replicate resamples SUBJECTS (patients) and keeps every stay's scores together.
#
# usage (RUN ONLY AFTER THE PROTOCOL IS FROZEN):
#   python3 src/06_auc_bootstrap.py --outcome hospital_death --window strict
#   python3 src/06_auc_bootstrap.py --selftest            # synthetic data only; safe to run any time
import argparse, json, math, os, subprocess, sys, time
import numpy as np
import pandas as pd

NB = 25                                              # score bins 0..24
SCORES = ["sofa1_community", "sofa1_ett_charted", "sofa2"]
COMPS = ["sofa1_community", "sofa1_ett_charted"]
HEADLINE = "sofa1_ett_charted"                       # headline comparator; the other is reported in full
ORDER = {"sex": ["Male", "Female"],
         "race": ["White", "Black", "Hispanic/Latino", "Asian", "Other", "Unknown/Declined"],
         "insurance": ["Medicare", "Medicaid", "Private", "Other/Missing"]}
PRIMARY_LEVELS = {"sex": ["Male", "Female"], "race": ["White", "Black", "Hispanic/Latino", "Asian", "Other"],
                  "insurance": ["Medicare", "Medicaid", "Private"]}
REF = {"sex": "Male", "race": "White", "insurance": "Medicare"}

def race_group(r):
    r = str(r).upper()
    if r in ("UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"):
        return "Unknown/Declined"
    for pref, g in [("WHITE", "White"), ("BLACK", "Black"), ("HISPANIC", "Hispanic/Latino"), ("ASIAN", "Asian")]:
        if r.startswith(pref):
            return g
    return "Other"

def auc_from_hist(pos, neg):
    P, N = pos.sum(-1), neg.sum(-1)
    below = np.cumsum(neg, -1) - neg
    with np.errstate(divide="ignore", invalid="ignore"):
        return (pos * (below + 0.5 * neg)).sum(-1) / (P * N)

def delong_paired(y, a, b):
    """Paired DeLong test for two correlated AUROCs (ignores patient clustering; cross-check only)."""
    y = np.asarray(y).astype(bool); m, n = y.sum(), (~y).sum()
    res = []
    for s in (np.asarray(a, float), np.asarray(b, float)):
        tz = pd.Series(s).rank(method="average").to_numpy()
        tx = pd.Series(s[y]).rank(method="average").to_numpy()
        ty = pd.Series(s[~y]).rank(method="average").to_numpy()
        auc = tz[y].sum() / m / n - (m + 1) / 2 / n
        res.append((auc, (tz[y] - tx) / n, 1 - (tz[~y] - ty) / m))
    sx = np.cov(np.vstack([res[0][1], res[1][1]])); sy = np.cov(np.vstack([res[0][2], res[1][2]]))
    cov = sx / m + sy / n
    d = res[0][0] - res[1][0]
    se = math.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    z = d / se if se > 0 else float("nan")
    return d, se, math.erfc(abs(z) / math.sqrt(2)) if se > 0 else float("nan")

def run_analysis(df, B, seed, outcome):
    """df columns: subject_id, y, sex, race, insurance, sofa1_community, sofa1_ett_charted, sofa2."""
    y = df["y"].to_numpy().astype(float)
    subj_idx, subj_uni = pd.factorize(df["subject_id"]); S = len(subj_uni)
    parts = list(ORDER)
    keys = {}
    for p in parts:
        codes = pd.Categorical(df[p], categories=ORDER[p]).codes.astype(np.int64)
        assert (codes >= 0).all(), f"unmapped level in {p}"
        keys[p] = {s: codes * NB + df[s].to_numpy().astype(np.int64) for s in SCORES}

    def aucs(w):
        out = {}
        posw, negw = w * y, w * (1 - y)
        for p in parts:
            L = len(ORDER[p])
            for s in SCORES:
                pos = np.bincount(keys[p][s], weights=posw, minlength=L * NB).reshape(L, NB)
                neg = np.bincount(keys[p][s], weights=negw, minlength=L * NB).reshape(L, NB)
                out[(p, s)] = auc_from_hist(pos, neg)
                if p == "sex":
                    out[("overall", s)] = auc_from_hist(pos.sum(0), neg.sum(0))
        return out

    point = aucs(np.ones(len(df)))
    rng = np.random.default_rng(seed)
    boot = {k: np.empty((B,) + np.shape(v)) for k, v in point.items()}
    t0 = time.time()
    for b in range(B):
        counts = np.bincount(rng.integers(0, S, S), minlength=S).astype(float)
        out = aucs(counts[subj_idx])
        for k, v in out.items():
            boot[k][b] = v
        if (b + 1) % 500 == 0:
            print(f"  bootstrap {b+1}/{B} ({time.time()-t0:,.0f}s)", flush=True)

    def ci(a):
        return np.nanpercentile(a, 2.5, axis=0), np.nanpercentile(a, 97.5, axis=0)
    def pv(a):
        a = a[~np.isnan(a)]
        return min(1.0, 2 * (min((a <= 0).sum(), (a >= 0).sum()) + 1) / (len(a) + 1))

    def cell(p, lvl):
        if p == "overall":
            sel = np.ones(len(df), bool); i = None
        else:
            sel = (df[p] == lvl).to_numpy(); i = ORDER[p].index(lvl)
        return sel, i
    def pick(arr, i):
        return arr if i is None else arr[..., i]

    rows = []
    levels = [("overall", "All stays")] + [(p, l) for p in parts for l in ORDER[p]]
    for p, lvl in levels:
        sel, i = cell(p, lvl)
        n, d = int(sel.sum()), int(y[sel].sum())
        prim = p == "overall" or lvl in PRIMARY_LEVELS.get(p, [])
        for s in SCORES:
            pt, bt = pick(point[(p if p == "overall" else p, s)], i), pick(boot[(p, s)], i)
            lo, hi = ci(bt)
            rows.append(dict(family="auroc", partition=p, level=lvl, ref="", score=s, comparator="", estimate=pt, lo=lo, hi=hi,
                             p_boot=np.nan, n=n, deaths=d, in_primary_set=prim))
        for c in COMPS:
            pt = pick(point[(p, "sofa2")], i) - pick(point[(p, c)], i)
            bt = pick(boot[(p, "sofa2")], i) - pick(boot[(p, c)], i)
            lo, hi = ci(bt)
            rows.append(dict(family="delta_auroc", partition=p, level=lvl, ref="", score="sofa2 - comparator", comparator=c,
                             estimate=pt, lo=lo, hi=hi, p_boot=pv(bt), n=n, deaths=d, in_primary_set=prim))
    for p in parts:                                    # subgroup gaps and change in gaps (vs reference level)
        ri = ORDER[p].index(REF[p])
        for lvl in PRIMARY_LEVELS[p]:
            if lvl == REF[p]:
                continue
            li = ORDER[p].index(lvl)
            sel = (df[p] == lvl).to_numpy(); n, d = int(sel.sum()), int(y[sel].sum())
            gap_pt = {s: point[(p, s)][ri] - point[(p, s)][li] for s in SCORES}
            gap_bt = {s: boot[(p, s)][:, ri] - boot[(p, s)][:, li] for s in SCORES}
            for s in SCORES:
                lo, hi = ci(gap_bt[s])
                rows.append(dict(family="gap", partition=p, level=lvl, ref=REF[p], score=s, comparator="",
                                 estimate=gap_pt[s], lo=lo, hi=hi, p_boot=pv(gap_bt[s]), n=n, deaths=d, in_primary_set=True))
            for c in COMPS:
                pt = gap_pt["sofa2"] - gap_pt[c]; bt = gap_bt["sofa2"] - gap_bt[c]
                lo, hi = ci(bt)
                rows.append(dict(family="delta_gap", partition=p, level=lvl, ref=REF[p], score="gap(sofa2) - gap(comparator)", comparator=c,
                                 estimate=pt, lo=lo, hi=hi, p_boot=pv(bt), n=n, deaths=d, in_primary_set=True))
    for c in COMPS:                                    # DeLong cross-check, overall only
        d_, se, pz = delong_paired(y, df["sofa2"].to_numpy(), df[c].to_numpy())
        rows.append(dict(family="delta_auroc_delong", partition="overall", level="All stays", ref="", score="sofa2 - comparator",
                         comparator=c, estimate=d_, lo=d_ - 1.96 * se, hi=d_ + 1.96 * se, p_boot=pz, n=len(df), deaths=int(y.sum()),
                         in_primary_set=True))
    res = pd.DataFrame(rows)
    res["outcome"] = outcome
    res["headline"] = res["comparator"].isin([HEADLINE, ""])
    res["tier"] = np.where(res["family"].isin(["delta_auroc", "delta_gap"]) & res["in_primary_set"] & (res["comparator"] == HEADLINE) & (outcome == "hospital_death"),
                           "primary", "co-primary/exploratory")
    return res

def fmt(r):
    return f"{r.estimate:.3f} ({r.lo:.3f} to {r.hi:.3f})"

def print_summary(res):
    pd.set_option("display.width", 220, "display.max_colwidth", 60)
    a = res[(res.family == "auroc") & (res.partition == "overall")]
    print("\nOverall AUROC:"); [print(f"  {r.score:18s} {fmt(r)}") for r in a.itertuples()]
    for c in COMPS:
        r = res[(res.family == "delta_auroc") & (res.partition == "overall") & (res.comparator == c)].iloc[0]
        dl = res[(res.family == "delta_auroc_delong") & (res.comparator == c)].iloc[0]
        print(f"  DeltaAUROC sofa2 - {c}: {fmt(r)}  bootstrap p={r.p_boot:.3f}; DeLong cross-check {dl.estimate:.4f}, p={dl.p_boot:.3f}")
    for c in COMPS:
        print(f"\nSubgroup AUROC (sofa2 | comparator {c} | delta):")
        t = res[(res.family.isin(["auroc", "delta_auroc"])) & (res.partition != "overall") & res.in_primary_set]
        for (p, l), g in t.groupby(["partition", "level"], sort=False):
            s2 = g[(g.family == "auroc") & (g.score == "sofa2")].iloc[0]; s1 = g[(g.family == "auroc") & (g.score == c)].iloc[0]
            dd = g[(g.family == "delta_auroc") & (g.comparator == c)].iloc[0]
            print(f"  {p:9s} {l:16s} n={dd.n:>6,} deaths={dd.deaths:>5,} | {s2.estimate:.3f} | {s1.estimate:.3f} | {fmt(dd)}")
        print(f"Change in subgroup AUROC gap (gap = reference - group), comparator {c}:")
        for r in res[(res.family == "delta_gap") & (res.comparator == c)].itertuples():
            print(f"  {r.partition:9s} {r.ref} - {r.level:16s} delta gap {fmt(r)}  p={r.p_boot:.3f}")

def selftest():
    print("SELF-TEST on synthetic data (no MIMIC data involved)")
    rng = np.random.default_rng(1); n = 30000
    subj = rng.integers(0, 22000, n)
    sex = rng.choice(["Male", "Female"], n, p=[.56, .44]); race = rng.choice(["White", "Black", "Hispanic/Latino", "Asian", "Other", "Unknown/Declined"], n, p=[.66, .11, .04, .03, .04, .12])
    ins = rng.choice(["Medicare", "Medicaid", "Private", "Other/Missing"], n, p=[.55, .15, .26, .04])
    sev = rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.1 * sev - 2.3)))).astype(int)
    noise = np.where(race == "Black", 1.6, 1.0)                    # injected: weaker discrimination in one group
    s1c = np.clip(np.round(4 + 2.2 * sev + rng.normal(0, 1.6, n) * noise), 0, 24).astype(int)
    s1e = np.clip(np.round(5 + 2.2 * sev + rng.normal(0, 1.8, n) * noise), 0, 24).astype(int)
    s2 = np.clip(np.round(4.5 + 2.2 * sev + rng.normal(0, 1.5, n) * noise), 0, 24).astype(int)
    df = pd.DataFrame(dict(subject_id=subj, y=y, sex=sex, race=race, insurance=ins, sofa1_community=s1c, sofa1_ett_charted=s1e, sofa2=s2))
    res = run_analysis(df, B=400, seed=7, outcome="hospital_death")
    # 1. histogram AUROC equals the rank-based (Mann-Whitney) AUROC
    def rank_auc(yv, sv):
        r = pd.Series(sv).rank().to_numpy(); m, k = yv.sum(), (1 - yv).sum()
        return (r[yv == 1].sum() - m * (m + 1) / 2) / (m * k)
    for s in SCORES:
        got = res[(res.family == "auroc") & (res.partition == "overall") & (res.score == s)].estimate.iloc[0]
        assert abs(got - rank_auc(y, df[s].to_numpy())) < 1e-12, s
    b = (df.race == "Black").to_numpy()
    got = res[(res.family == "auroc") & (res.partition == "race") & (res.level == "Black") & (res.score == "sofa2")].estimate.iloc[0]
    assert abs(got - rank_auc(y[b], s2[b])) < 1e-12
    # 2. identical scores -> zero delta with zero-width CI
    df2 = df.assign(sofa1_community=df["sofa2"])
    r2 = run_analysis(df2, B=100, seed=3, outcome="hospital_death")
    z = r2[(r2.family == "delta_auroc") & (r2.partition == "overall") & (r2.comparator == "sofa1_community")].iloc[0]
    assert z.estimate == 0 and z.lo == 0 and z.hi == 0
    # 3. injected weaker discrimination shows up as a positive reference-minus-group gap
    g = res[(res.family == "gap") & (res.partition == "race") & (res.level == "Black") & (res.score == "sofa2")].iloc[0]
    assert g.estimate > 0.02, g.estimate
    # 4. bootstrap SE of the overall delta is close to the DeLong SE
    o = res[(res.family == "delta_auroc") & (res.partition == "overall") & (res.comparator == "sofa1_ett_charted")].iloc[0]
    dl = res[(res.family == "delta_auroc_delong") & (res.comparator == "sofa1_ett_charted")].iloc[0]
    se_boot, se_dl = (o.hi - o.lo) / 3.92, (dl.hi - dl.lo) / 3.92
    assert 0.6 < se_boot / se_dl < 1.6, (se_boot, se_dl)
    print("all checks passed: histogram AUROC == rank AUROC; identical scores give delta 0; injected gap detected; "
          f"bootstrap SE {se_boot:.4f} vs DeLong SE {se_dl:.4f}")
    print_summary(res)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="hospital_death", choices=["hospital_death", "icu_death_strict"])
    ap.add_argument("--window", default="strict")
    ap.add_argument("--sofa2-tag", default=None, help="e.g. strict_ne_half (default = window)")
    ap.add_argument("--sofa2-col", default="sofa2_total")
    ap.add_argument("--complete-case", action="store_true")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20251007)
    ap.add_argument("--age-lt", type=int, default=None, help="keep stays with age below N (prespecified insurance sensitivity)")
    ap.add_argument("--careunit", default=None, help="keep stays whose first care unit contains this text, e.g. (MICU)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    t2 = a.sofa2_tag or (a.window + "_sed_broad")      # headline SOFA-2 = broad sedation rule
    coh = pd.read_csv("outputs/tables/cohort_PRIVATE_do_not_share.csv").set_index("stay_id")
    f = lambda name: pd.read_csv(f"outputs/tables/{name}_PRIVATE.csv").set_index("stay_id")
    s1c, s1e, s2 = f(f"sofa1_{a.window}"), f(f"sofa1_{a.window}_ett_charted"), f(f"sofa2_{t2}")
    df = pd.DataFrame({"subject_id": coh["subject_id"], "y": coh[a.outcome].astype(int),
                       "sex": coh["gender"].map({"M": "Male", "F": "Female"}), "race": coh["race"].map(race_group),
                       "insurance": coh["insurance"].fillna("Other/Missing").where(coh["insurance"].isin(["Medicare", "Medicaid", "Private"]), "Other/Missing"),
                       "sofa1_community": s1c["sofa1_total"], "sofa1_ett_charted": s1e["sofa1_total"], "sofa2": s2[a.sofa2_col],
                       "age": coh["age"], "careunit": coh["first_careunit"].fillna("Unknown")})
    assert df.notna().all().all(), "unexpected missing values"
    if a.age_lt is not None:
        df = df[df["age"] < a.age_lt]
    if a.careunit:
        df = df[df["careunit"].str.contains(a.careunit, case=False, regex=False)]
    if a.complete_case:
        cc = s1c["complete_case"] & s1e["complete_case"] & s2["complete_case"]
        df = df[cc.reindex(df.index).fillna(False)]
    print(f"N stays = {len(df):,}; subjects = {df.subject_id.nunique():,}; deaths ({a.outcome}) = {int(df.y.sum()):,}; "
          f"window = {a.window}; sofa2 = {t2}/{a.sofa2_col}; B = {a.B}; seed = {a.seed}")
    res = run_analysis(df.reset_index(drop=True), a.B, a.seed, a.outcome)
    tag = f"{a.outcome}_{a.window}" + ("" if t2 == a.window + "_sed_broad" else f"_s2-{t2}") + ("" if a.sofa2_col == "sofa2_total" else f"_{a.sofa2_col}") + (f"_age-lt{a.age_lt}" if a.age_lt is not None else "") + (("_cu-" + "".join(ch for ch in a.careunit if ch.isalnum())) if a.careunit else "") + ("_cc" if a.complete_case else "")
    res.to_csv(f"outputs/tables/auc_results_{tag}.csv", index=False)
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git = "not a git repository"
    json.dump({"outcome": a.outcome, "window": a.window, "sofa2_tag": t2, "sofa2_col": a.sofa2_col, "complete_case": a.complete_case,
               "B": a.B, "seed": a.seed, "n_stays": len(df), "n_subjects": int(df.subject_id.nunique()), "deaths": int(df.y.sum()),
               "numpy": np.__version__, "pandas": pd.__version__, "python": sys.version.split()[0], "git_commit": git,
               "bootstrap": "paired subject-cluster resampling, percentile 95% CI, two-sided bootstrap p"},
              open(f"outputs/tables/analysis_manifest_{tag}.json", "w"), indent=2)
    print_summary(res)
    print(f"\nSaved outputs/tables/auc_results_{tag}.csv and analysis_manifest_{tag}.json")

if __name__ == "__main__":
    main()
