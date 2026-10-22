import pandas as pd, glob, os
b, a = "outputs_before_fix/tables/", "outputs/tables/"
for pat in ["auc_results_*.csv", "calib_results_*.csv", "domain_decomposition_*hospital*.csv"]:
    for f in sorted(glob.glob(b + pat)):
        n = os.path.basename(f); x, y = pd.read_csv(f), pd.read_csv(a + n)
        if x.shape != y.shape or list(x.columns) != list(y.columns): continue
        keys = [c for c in x.columns if x[c].dtype.kind in "Ob"]
        num = [c for c in x.columns if x[c].dtype.kind == "f" and "p_boot" not in c and c != "p"]
        out = []
        for c in num:
            d = (x[c].round(3) != y[c].round(3)) & ~(x[c].isna() & y[c].isna())
            for i in x.index[d]:
                k = "|".join(str(x.loc[i, q]) for q in keys if str(x.loc[i, q]) not in ("nan", ""))
                out.append(f"  {k} [{c}] {x.loc[i, c]:.4f} -> {y.loc[i, c]:.4f}")
        if out:
            print("==", n)
            print("\n".join(out))
