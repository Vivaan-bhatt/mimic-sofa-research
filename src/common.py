# common.py -- shared loading, window and plausible-range rules used by BOTH score calculators,
# so SOFA-1 and SOFA-2 see exactly the same cleaned inputs.
import numpy as np
import pandas as pd

EX = "outputs/extract"
H = pd.Timedelta(hours=1)

# INTERIM plausible ranges (ours, wide). Replace with SOFA-2 Supplement 2 eTable 2 when obtained.
RANGES = {
    "gcs_eye": (1, 4), "gcs_verbal": (0, 5), "gcs_motor": (1, 6),   # verbal 0 = intubated (ETT)
    "fio2": (21, 100), "pao2": (20, 700), "spo2": (1, 100), "map": (10, 300),
    "platelets": (1, 2000), "bilirubin": (0, 100), "creatinine": (0.1, 30),
    "potassium": (1.5, 12), "ph": (6.5, 8.0), "bicarbonate": (2, 60), "weight": (20, 400),
    "uo_record": (0, 5000),
    "norepinephrine": (0, 5), "epinephrine": (0, 5), "dopamine": (0, 50),
}
# window offsets in hours relative to ICU admission: half-open [lo, hi)
WINDOWS = {"strict": (0, 24), "lookback6": (-6, 24)}

def load_cohort():
    return pd.read_csv("outputs/tables/cohort_PRIVATE_do_not_share.csv",
                       parse_dates=["intime", "outtime", "window_end"])

def read_extract(name, time_cols=(), num_cols=(), int_cols=("stay_id", "itemid")):
    df = pd.read_csv(f"{EX}/{name}.csv.gz", dtype=str, keep_default_na=False)
    for c in int_cols:
        if c in df:
            df[c] = df[c].astype("int64")
    for c in time_cols:
        df[c] = pd.to_datetime(df[c])
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def in_window(df, coh, mode, time_col=None, start_col=None, end_col=None):
    """Point events: lo <= t < hi. Interval events: overlap with [lo, hi)."""
    lo_h, hi_h = WINDOWS[mode]
    intime = df["stay_id"].map(coh.set_index("stay_id")["intime"])
    lo, hi = intime + lo_h * H, intime + hi_h * H
    if time_col:
        return (df[time_col] >= lo) & (df[time_col] < hi)
    return (df[start_col] < hi) & (df[end_col] > lo)

def rng(s, key):
    lo, hi = RANGES[key]
    return s.where(s.between(lo, hi))

def fio2_percent(v):
    v = v.copy()
    v = v.where(~(v <= 1), v * 100)       # fractions -> percent
    return rng(v, "fio2")

# ---------------------------------------------------------------------------------------------
# helpers added for SOFA-2 (04_sofa2.py)
MIN = pd.Timedelta(minutes=1)

def gcs_pivot(chart):
    """One row per (stay, charttime) with eye/verbal/motor; verbal 0 = 'No Response-ETT'.
    *_f columns carry components forward up to 6 h."""
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
    return piv[piv[["eye", "verbal", "motor"]].notna().any(axis=1)].reset_index(drop=True)

def add_union(df, keys):
    """Merge overlapping/touching intervals within keys; adds gs, ge (union start/end) and dur_h."""
    df = df.sort_values(keys + ["starttime"]).copy()
    df["_cmax"] = df.groupby(keys)["endtime"].cummax()
    prev = df.groupby(keys)["_cmax"].shift()
    df["_grp"] = (prev.isna() | (df["starttime"] > prev)).cumsum()
    g = df.groupby("_grp").agg(gs=("starttime", "min"), ge=("endtime", "max"))
    df = df.join(g, on="_grp")
    df["dur_h"] = (df["ge"] - df["gs"]) / H
    return df.drop(columns=["_cmax", "_grp"])

def _aligned(left, merged_col, ok):
    return pd.Series(ok.to_numpy(), index=merged_col.to_numpy()).sort_index().to_numpy()

def covered(df, iv):
    """True where charttime lies inside an interval (stay_id, starttime, endtime) of iv."""
    if len(df) == 0 or len(iv) == 0:
        return np.zeros(len(df), dtype=bool)
    left = df[["stay_id", "charttime"]].copy(); left["_i"] = np.arange(len(left))
    left = left.sort_values("charttime")
    r = iv[["stay_id", "starttime", "endtime"]].sort_values("starttime")
    m = pd.merge_asof(left, r, left_on="charttime", right_on="starttime", by="stay_id", direction="backward")
    ok = m["endtime"].notna() & (m["charttime"] <= m["endtime"])
    return _aligned(left, m["_i"], ok)

def asof_has(df, ev, tol):
    """True where an event (stay_id, t) exists within `tol` before charttime."""
    if len(df) == 0 or len(ev) == 0:
        return np.zeros(len(df), dtype=bool)
    left = df[["stay_id", "charttime"]].copy(); left["_i"] = np.arange(len(left))
    left = left.sort_values("charttime")
    m = pd.merge_asof(left, ev[["stay_id", "t"]].sort_values("t"), left_on="charttime", right_on="t",
                      by="stay_id", direction="backward", tolerance=tol)
    return _aligned(left, m["_i"], m["t"].notna())

def asof_value(df, ev, col, tol):
    """Latest value of ev[col] (stay_id, t, col) within `tol` before charttime; NaN if none."""
    if len(df) == 0 or len(ev) == 0:
        return np.full(len(df), np.nan, dtype=object)
    left = df[["stay_id", "charttime"]].copy(); left["_i"] = np.arange(len(left))
    left = left.sort_values("charttime")
    m = pd.merge_asof(left, ev[["stay_id", "t", col]].sort_values("t"), left_on="charttime", right_on="t",
                      by="stay_id", direction="backward", tolerance=tol)
    return pd.Series(m[col].to_numpy(), index=m["_i"].to_numpy()).sort_index().to_numpy()
