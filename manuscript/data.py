import pandas as pd, numpy as np, json, os, glob
RES = os.environ.get("RESULTS_DIR") or ("../outputs/tables" if os.path.isdir("../outputs/tables") else "share")
S = RES
COHORT = os.environ.get("COHORT_FILE") or os.path.join(RES, "cohort_PRIVATE_do_not_share.csv")
def rd(name): return pd.read_csv(f"{S}/{name}.csv")
H = rd("auc_results_hospital_death_strict"); I = rd("auc_results_icu_death_strict_strict")
CH = rd("calib_results_hospital_death_strict"); CB = rd("calib_bands_hospital_death_strict")
DD = rd("domain_decomposition_hospital_death_strict"); DDI = rd("domain_decomposition_icu_death_strict_strict")
FLOW = rd("cohort_flow"); SPO2 = rd("spo2_vs_pao2_by_group"); MISS = rd("audit_missingness_by_subgroup")
SC = {"sofa1_community": "SOFA-1 (community)", "sofa1_ett_charted": "SOFA-1 (as charted)", "sofa2": "SOFA-2"}
COMPS = ["sofa1_community", "sofa1_ett_charted"]
SUBS = {"sex": ["Male", "Female"], "race": ["White", "Black", "Hispanic/Latino", "Asian", "Other"], "insurance": ["Medicare", "Medicaid", "Private"]}
REF = {"sex": "Male", "race": "White", "insurance": "Medicare"}
MINUS = "\u2212"
def sg(x, d=3, signed=False):
    s = f"{abs(x):.{d}f}"
    if x < 0 and float(s) != 0: return MINUS + s
    return ("+" + s) if signed and float(s) != 0 else s
def ci(r, d=3, signed=False):
    return f"{sg(r.estimate, d, signed)} ({sg(r.lo, d, signed)} to {sg(r.hi, d, signed)})"
def pf(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"
def get(df, family, partition, level=None, comp=None, score=None):
    m = (df.family == family) & (df.partition == partition)
    if level is not None: m &= df.level == level
    if comp is not None: m &= df.comparator == comp
    if score is not None: m &= df.score == score
    r = df[m]
    assert len(r) == 1, (family, partition, level, comp, score, len(r))
    return r.iloc[0]
def dist_stats(f, col):
    d = rd(f).set_index(col)["n"].reindex(range(25), fill_value=0); n = d.sum(); x = d.index.to_numpy(); w = d.to_numpy()
    cum = np.cumsum(w) / n
    q = lambda p: int(x[np.searchsorted(cum, p)])
    return dict(n=n, mean=(x * w).sum() / n, median=q(0.5), q1=q(0.25), q3=q(0.75), dist=w / n)
D = {"sofa1_community": dist_stats("sofa1_strict_distribution", "sofa1_total"),
     "sofa1_ett_charted": dist_stats("sofa1_strict_ett_charted_distribution", "sofa1_total"),
     "sofa2": dist_stats("sofa2_strict_sed_broad_distribution", "sofa2_total"),
     "sofa2_narrow": dist_stats("sofa2_strict_distribution", "sofa2_total")}
MAN = json.load(open(f"{S}/analysis_manifest_hospital_death_strict.json"))

DS1 = rd("sofa1_strict_domain_summary"); DS2B = rd("sofa2_strict_sed_broad_domain_summary"); DS2N = rd("sofa2_strict_domain_summary")
def obs(ds, dom): return 100 - float(ds[ds.domain == dom].pct_missing.iloc[0])
