import os, sys, time
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

base = os.path.expanduser(sys.argv[1])
table = sys.argv[2]
MIN_COUNT = 20          # text values seen fewer times than this are hidden
CHUNK = 2_000_000

CFG = {
 "outputevents": dict(
    path="icu/outputevents.csv.gz",
    ids=[226557,226558,226559,226560,226561,226563,226564,226565,226566,226567,226584,226627,226631,226713,227488,227489],
    usecols=["stay_id","itemid","value","valueuom"], entity="stay_id",
    num_cols=["value"], cat_cols=["valueuom"], text_col=None, dict="items"),
 "procedureevents": dict(
    path="icu/procedureevents.csv.gz",
    ids=[225792,225794,224385,227194,225441,225802,225803,225809,225955,225805,229529,229530,224272,228169,224270],
    usecols=["stay_id","itemid","value","valueuom","statusdescription"], entity="stay_id",
    num_cols=["value"], cat_cols=["valueuom","statusdescription"], text_col=None, dict="items"),
 "inputevents": dict(
    path="icu/inputevents.csv.gz",
    ids=[221289,229617,221653,221662,221749,229630,229631,229632,221906,221986,222315,229709,229764,
         222168,221668,225150,229420,221744,225942,225972,221385,221712,221824],
    usecols=["stay_id","itemid","amount","amountuom","rate","rateuom","ordercategorydescription","patientweight","statusdescription"],
    entity="stay_id", num_cols=["amount","rate","patientweight"],
    cat_cols=["amountuom","rateuom","ordercategorydescription","statusdescription"], text_col=None, dict="items"),
 "labevents": dict(
    path="hosp/labevents.csv.gz",
    ids=[50885,53089,50912,52546,51265,53189,50821,50816,50817,52033,50820,50882,50971,52610],
    usecols=["subject_id","itemid","value","valuenum","valueuom"], entity="subject_id",
    num_cols=["valuenum"], cat_cols=["valueuom"], text_col="value", numeric_companion="valuenum", dict="labs"),
 "chartevents": dict(
    path="icu/chartevents.csv.gz",
    ids=[220739,223900,223901,223835,220277,220052,220181,225312,
         223848,223849,229314,230045,226732,225949,227577,227583,
         224660,229270,228193,227980,228154,229671,220125,220128,229254,
         226512,224639,226531,228096,228332,225126,227290],
    usecols=["stay_id","itemid","value","valuenum","valueuom"], entity="stay_id",
    num_cols=["valuenum"], cat_cols=["valueuom"], text_col="value", numeric_companion="valuenum", dict="items"),
}
cfg = CFG[table]
idset = set(str(i) for i in cfg["ids"])

class Acc:
    def __init__(self):
        self.n = 0
        self.ents = set()
        self.cats = defaultdict(Counter)
        self.num = defaultdict(lambda: dict(n=0, neg=0, zero=0, mn=np.inf, mx=-np.inf))
        self.samp = defaultdict(list)

A = defaultdict(Acc)
rng = np.random.default_rng(0)
t0 = time.time()
rows_seen = 0

reader = pd.read_csv(os.path.join(base, cfg["path"]), usecols=cfg["usecols"],
                     dtype=str, chunksize=CHUNK)
for k, chunk in enumerate(reader):
    rows_seen += len(chunk)
    sub = chunk[chunk["itemid"].isin(idset)]
    if len(sub):
        for iid, g in sub.groupby("itemid"):
            a = A[int(iid)]
            a.n += len(g)
            a.ents.update(g[cfg["entity"]].dropna().unique())
            for c in cfg["cat_cols"]:
                a.cats[c].update(g[c].fillna("NA").value_counts().to_dict())
            if cfg["text_col"]:
                nonnum = g[pd.to_numeric(g[cfg["numeric_companion"]], errors="coerce").isna()]
                a.cats["text_value"].update(nonnum[cfg["text_col"]].fillna("NA").value_counts().to_dict())
            for c in cfg["num_cols"]:
                v = pd.to_numeric(g[c], errors="coerce").dropna().to_numpy(dtype="float64")
                if len(v):
                    s = a.num[c]
                    s["n"] += len(v); s["neg"] += int((v < 0).sum()); s["zero"] += int((v == 0).sum())
                    s["mn"] = min(s["mn"], v.min()); s["mx"] = max(s["mx"], v.max())
                    a.samp[c].append(v if len(v) <= 20000 else rng.choice(v, 20000, replace=False))
            for c in list(a.cats):
                if len(a.cats[c]) > 3000:
                    a.cats[c] = Counter(dict(a.cats[c].most_common(1000)))
    print(f"chunk {k+1}: {rows_seen:,} rows read, {time.time()-t0:,.0f}s elapsed", flush=True)

if cfg["dict"] == "items":
    d = pd.read_csv(os.path.join(base, "icu", "d_items.csv.gz")).set_index("itemid")["label"].to_dict()
else:
    d = pd.read_csv(os.path.join(base, "hosp", "d_labitems.csv.gz")).set_index("itemid")["label"].to_dict()

out = []
for iid in cfg["ids"]:
    a = A.get(iid, Acc())
    row = {"table": table, "itemid": iid, "label": d.get(iid, "?"), "n_rows": a.n, "n_entities": len(a.ents)}
    for c in cfg["num_cols"]:
        s = a.num[c] if c in a.num else None
        if s and s["n"]:
            samp = np.concatenate(a.samp[c])
            q = np.percentile(samp, [1, 25, 50, 75, 99])
            row.update({f"{c}_n": s["n"], f"{c}_min": s["mn"], f"{c}_p1": q[0], f"{c}_p25": q[1],
                        f"{c}_median": q[2], f"{c}_p75": q[3], f"{c}_p99": q[4], f"{c}_max": s["mx"],
                        f"{c}_n_negative": s["neg"], f"{c}_n_zero": s["zero"]})
        else:
            row[f"{c}_n"] = 0
    for c, cnt in a.cats.items():
        keep = [(v, n) for v, n in cnt.most_common(15) if n >= MIN_COUNT]
        row[f"{c}_top"] = "; ".join(f"{v}: {n}" for v, n in keep)
    out.append(row)

res = pd.DataFrame(out)
res.to_csv(f"sofa_counts_{table}.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 120)
print(res.to_string(index=False))
print(f"\nSaved sofa_counts_{table}.csv ({time.time()-t0:,.0f}s)")
