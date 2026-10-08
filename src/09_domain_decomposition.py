# 09_domain_decomposition.py -- which SOFA-2 change drives the AUROC gain and the change in subgroup gaps?
# For each domain, build a HYBRID score = SOFA-1 total with that ONE domain replaced by its SOFA-2 version, then run the
# same paired patient-cluster bootstrap against the unmodified SOFA-1 (same comparator). Exploratory by design.
#   python3 src/09_domain_decomposition.py --outcome hospital_death --window strict
#   python3 src/09_domain_decomposition.py --selftest
import argparse, importlib.util, os, sys
import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("auc06", os.path.join(here, "06_auc_bootstrap.py"))
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
DOMS = [("brain", "cns_pts", "brain_pts"), ("respiratory", "resp_pts", "resp_pts"), ("cardiovascular", "cv_pts", "cv_pts"),
        ("liver", "liver_pts", "liver_pts"), ("kidney", "renal_pts", "renal_pts"), ("hemostasis", "coag_pts", "hemo_pts")]

def hybrid_scores(s1_by_comp, s2):
    out = {}
    for comp, s1 in s1_by_comp.items():
        for name, d1, d2 in DOMS:
            out[(comp, name)] = (s1["sofa1_total"] - s1[d1].fillna(0) + s2[d2].fillna(0)).astype(int)
    return out

def decompose(df, hyb, B, seed, outcome):
    rows = []
    for (comp, name), h in hyb.items():
        dfh = df.copy(); dfh["sofa2"] = h.to_numpy()
        res = A.run_analysis(dfh, B, seed, outcome)
        keep = res[((res.family == "delta_auroc") | (res.family == "delta_gap")) & (res.comparator == comp)].copy()
        keep["swapped_domain"], keep["sofa1_comparator"] = name, comp
        rows.append(keep)
    return pd.concat(rows, ignore_index=True)

def show(r):
    return f"{r.estimate:+.3f} ({r.lo:+.3f} to {r.hi:+.3f})"

def print_table(full, tot):
    for comp in A.COMPS:
        print(f"\nSOFA-1 comparator: {comp}   (all-domain SOFA-2 vs this comparator: overall dAUROC {tot[comp]})")
        print(f"  {'domain swapped':15s} {'overall dAUROC':26s} {'gap change: sex M-F':26s} {'White-Black':26s} {'Medicare-Private':26s}")
        for name, _, _ in DOMS:
            g = full[(full.sofa1_comparator == comp) & (full.swapped_domain == name)]
            o = g[(g.family == "delta_auroc") & (g.partition == "overall")].iloc[0]
            def gap(p, lvl):
                return show(g[(g.family == "delta_gap") & (g.partition == p) & (g.level == lvl)].iloc[0])
            print(f"  {name:15s} {show(o):26s} {gap('sex', 'Female'):26s} {gap('race', 'Black'):26s} {gap('insurance', 'Private'):26s}")

def selftest():
    print("SELF-TEST on synthetic data (no MIMIC data involved)")
    rng = np.random.default_rng(4); n = 25000
    sev = rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.2 * sev - 2.2)))).astype(int)
    d1 = {c: np.clip(np.round(1 + 0.6 * sev + rng.normal(0, 1.0, n)), 0, 4) for _, c, _ in DOMS}
    d2 = {c2: d1[c1].copy() for _, c1, c2 in DOMS}
    d2["brain_pts"] = np.clip(np.round(1 + 1.0 * sev + rng.normal(0, 0.5, n)), 0, 4)          # only the brain domain differs
    idx = pd.RangeIndex(n, name="stay_id")
    s1 = pd.DataFrame({c: v for c, v in d1.items()}, index=idx); s1["sofa1_total"] = sum(v for v in d1.values()).astype(int)
    s2 = pd.DataFrame({c: v for c, v in d2.items()}, index=idx); s2["sofa2_total"] = sum(v for v in d2.values()).astype(int)
    hyb = hybrid_scores({"sofa1_community": s1, "sofa1_ett_charted": s1}, s2)
    full_swap = s1["sofa1_total"] - sum(s1[c1].fillna(0) for _, c1, _ in DOMS) + sum(s2[c2].fillna(0) for _, _, c2 in DOMS)
    assert (full_swap == s2["sofa2_total"]).all(), "swapping every domain must reproduce SOFA-2 exactly"
    sex = rng.choice(["Male", "Female"], n); race = rng.choice(A.ORDER["race"], n, p=[.66, .11, .04, .03, .04, .12])
    ins = rng.choice(A.ORDER["insurance"], n, p=[.55, .15, .26, .04])
    df = pd.DataFrame(dict(subject_id=rng.integers(0, 20000, n), y=y, sex=sex, race=race, insurance=ins,
                           sofa1_community=s1["sofa1_total"].to_numpy(), sofa1_ett_charted=s1["sofa1_total"].to_numpy(), sofa2=s2["sofa2_total"].to_numpy()))
    sub = {k: v for k, v in hyb.items() if k[0] == "sofa1_ett_charted" and k[1] in ("brain", "kidney")}
    full = decompose(df, sub, B=150, seed=3, outcome="hospital_death")
    get = lambda nm: full[(full.swapped_domain == nm) & (full.family == "delta_auroc") & (full.partition == "overall")].iloc[0]
    assert get("brain").estimate > 0.01 and get("brain").lo > 0, get("brain")
    assert abs(get("kidney").estimate) < 1e-12, "a domain that did not change must give a zero effect"
    print(f"all checks passed: all-domain swap reproduces SOFA-2 exactly; brain-only change gives dAUROC {get('brain').estimate:+.3f}; unchanged domain gives 0")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="hospital_death", choices=["hospital_death", "icu_death_strict"])
    ap.add_argument("--window", default="strict"); ap.add_argument("--sofa2-tag", default=None)
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--seed", type=int, default=20251007)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    t2 = a.sofa2_tag or (a.window + "_sed_broad")
    coh = pd.read_csv("outputs/tables/cohort_PRIVATE_do_not_share.csv").set_index("stay_id")
    f = lambda name: pd.read_csv(f"outputs/tables/{name}_PRIVATE.csv").set_index("stay_id")
    s1c, s1e, s2 = f(f"sofa1_{a.window}"), f(f"sofa1_{a.window}_ett_charted"), f(f"sofa2_{t2}")
    df = pd.DataFrame({"subject_id": coh["subject_id"], "y": coh[a.outcome].astype(int),
                       "sex": coh["gender"].map({"M": "Male", "F": "Female"}), "race": coh["race"].map(A.race_group),
                       "insurance": coh["insurance"].fillna("Other/Missing").where(coh["insurance"].isin(["Medicare", "Medicaid", "Private"]), "Other/Missing"),
                       "sofa1_community": s1c["sofa1_total"], "sofa1_ett_charted": s1e["sofa1_total"], "sofa2": s2["sofa2_total"]})
    assert df.notna().all().all()
    chk = (s1e["sofa1_total"] - s1e[[d[1] for d in DOMS]].fillna(0).sum(axis=1)).abs().max()
    assert chk == 0, "SOFA-1 domains must sum to the SOFA-1 total"
    hyb = hybrid_scores({"sofa1_community": s1c, "sofa1_ett_charted": s1e}, s2.reindex(s1c.index))
    full = decompose(df.reset_index(drop=True), {k: v.reindex(df.index) for k, v in hyb.items()}, a.B, a.seed, a.outcome)
    tag = f"{a.outcome}_{a.window}"
    full.to_csv(f"outputs/tables/domain_decomposition_{tag}.csv", index=False)
    tot = {}
    for c in A.COMPS:
        r = A.run_analysis(df.reset_index(drop=True), 300, a.seed, a.outcome)
        o = r[(r.family == "delta_auroc") & (r.partition == "overall") & (r.comparator == c)].iloc[0]
        tot[c] = f"{o.estimate:+.3f}"
    print(f"N stays = {len(df):,}; deaths = {int(df.y.sum()):,}; B = {a.B}")
    print_table(full, tot)
    print(f"\nSaved outputs/tables/domain_decomposition_{tag}.csv")

if __name__ == "__main__":
    main()
