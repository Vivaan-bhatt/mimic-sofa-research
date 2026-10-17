import os; os.makedirs("fig", exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from data import *
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COL = {"sofa1_community": "#8c8c8c", "sofa1_ett_charted": "#d9822b", "sofa2": "#1f5fa8"}

# ---- Fig 1: cohort flow
fig, ax = plt.subplots(figsize=(6.2, 4.6)); ax.axis("off")
labels = [r.step.split(". ", 1)[1] for r in FLOW.itertuples()]
ys = [0.88, 0.70, 0.52, 0.34, 0.16]
for (r, y, lab) in zip(FLOW.itertuples(), ys, labels):
    ax.add_patch(FancyBboxPatch((0.12, y - 0.065), 0.76, 0.13, boxstyle="round,pad=0.01", fc="#eef3fa", ec="#1f5fa8", lw=1))
    ax.text(0.5, y + 0.02, lab, ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(0.5, y - 0.03, f"{r.icu_stays:,} ICU stays | {r.hospitalizations:,} hospitalizations | {r.patients:,} patients", ha="center", va="center", fontsize=8.5)
for y in ys[:-1]:
    ax.annotate("", xy=(0.5, y - 0.075), xytext=(0.5, y - 0.105), arrowprops=dict(arrowstyle="-|>", color="#555"))
ax.text(0.5, 0.02, f"Analytic cohort: {MAN['n_stays']:,} stays from {MAN['n_subjects']:,} patients; {MAN['deaths']:,} in-hospital deaths", ha="center", fontsize=9, style="italic")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); fig.savefig("fig/fig1_flow.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- Fig 2: score distributions
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for a, k in zip(axs, SC):
    d = D[k]; a.bar(range(25), 100 * d["dist"], color=COL[k], width=0.85)
    a.set_title(SC[k], fontsize=9); a.set_xlabel("Total score (first 24 h)")
    a.text(0.97, 0.92, f"median {d['median']} (IQR {d['q1']}\u2013{d['q3']})\nmean {d['mean']:.2f}", transform=a.transAxes, ha="right", va="top", fontsize=7.5)
axs[0].set_ylabel("% of stays"); fig.tight_layout(); fig.savefig("fig/fig2_distributions.png", dpi=200); plt.close()

# ---- forest helper
def forest(rows, title, xlabel, fname, xlim=None, note=None):
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(rows) + 1.3))
    for i, (lab, items) in enumerate(rows):
        y = len(rows) - i
        for j, (c, r) in enumerate(items):
            off = (0.14 if j == 0 else -0.14)
            ax.errorbar(r.estimate, y + off, xerr=[[r.estimate - r.lo], [r.hi - r.estimate]], fmt="o", ms=4, color=COL[c], capsize=2, lw=1.1)
    ax.axvline(0, color="#444", lw=0.8, ls="--")
    ax.set_yticks(range(1, len(rows) + 1)); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8)
    ax.set_xlabel(xlabel); ax.set_title(title, fontsize=9, loc="left")
    if xlim: ax.set_xlim(*xlim)
    for c, lab in [("sofa1_ett_charted", "vs SOFA-1 as charted (headline)"), ("sofa1_community", "vs SOFA-1 community")]:
        ax.plot([], [], "o", color=COL[c], label=lab)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2); fig.tight_layout(); fig.savefig(fname, dpi=200, bbox_inches="tight"); plt.close()

# ---- Fig 3: delta AUROC forest
rows = [("All stays", [(c, get(H, "delta_auroc", "overall", "All stays", c)) for c in ["sofa1_ett_charted", "sofa1_community"]])]
for p, lv in SUBS.items():
    for l in lv:
        rows.append((f"{p.capitalize()}: {l}", [(c, get(H, "delta_auroc", p, l, c)) for c in ["sofa1_ett_charted", "sofa1_community"]]))
forest(rows, "Change in AUROC for in-hospital mortality (SOFA-2 minus SOFA-1)", "\u0394AUROC (95% paired-bootstrap CI)", "fig/fig3_delta_auroc.png", xlim=(-0.02, 0.075))

# ---- Fig 4: delta gap forest
rows = []
for p, lv in SUBS.items():
    for l in lv:
        if l == REF[p]: continue
        rows.append((f"{REF[p]} \u2212 {l}", [(c, get(H, "delta_gap", p, l, c)) for c in ["sofa1_ett_charted", "sofa1_community"]]))
forest(rows, "Change in subgroup AUROC gap (gap = reference \u2212 group); SOFA-2 gap minus SOFA-1 gap", "\u0394gap (95% paired-bootstrap CI)", "fig/fig4_delta_gap.png", xlim=(-0.05, 0.045))

# ---- Fig 5: calibration (observed mortality by band)
bands = list(CB.band.unique())
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
x = np.arange(len(bands))
for j, c in enumerate(SC):
    d = CB[(CB.partition == "overall") & (CB.score == c)].set_index("band").reindex(bands)
    axs[0].errorbar(x + (j - 1) * 0.14, 100 * d.observed, yerr=[100 * (d.observed - d.wilson_lo), 100 * (d.wilson_hi - d.observed)], fmt="o-", ms=3, lw=0.8, capsize=1.5, color=COL[c], label=SC[c])
axs[0].set_title("A. Overall, by score", fontsize=9, loc="left"); axs[0].legend(fontsize=7, frameon=False)
for j, (l, colr) in enumerate([("Medicare", "#b03a2e"), ("Medicaid", "#2e8b57"), ("Private", "#1f5fa8")]):
    d = CB[(CB.partition == "insurance") & (CB.level == l) & (CB.score == "sofa2")].set_index("band").reindex(bands)
    axs[1].errorbar(x + (j - 1) * 0.14, 100 * d.observed, yerr=[100 * (d.observed - d.wilson_lo), 100 * (d.wilson_hi - d.observed)], fmt="o-", ms=3, lw=0.8, capsize=1.5, color=colr, label=l)
axs[1].set_title("B. SOFA-2 by insurance", fontsize=9, loc="left"); axs[1].legend(fontsize=7, frameon=False)
for a in axs:
    a.set_xticks(x); a.set_xticklabels(bands, rotation=60, fontsize=7); a.set_xlabel("Score band")
axs[0].set_ylabel("Observed in-hospital mortality (%)"); fig.tight_layout(); fig.savefig("fig/fig5_calibration.png", dpi=200); plt.close()

# ---- Fig 6: decomposition (as-charted comparator)
dom_order = ["respiratory", "respiratory (P/F only)", "brain", "cardiovascular", "kidney", "hemostasis", "liver"]
def sel(dd, family, partition, level=None):
    out = []
    for d in dom_order:
        m = (dd.sofa1_comparator == "sofa1_ett_charted") & (dd.swapped_domain == d) & (dd.family == family) & (dd.partition == partition)
        if level: m &= dd.level == level
        out.append(dd[m].iloc[0])
    return out
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
for a, (ttl, fam, part, lvl) in zip(axs, [("A. \u0394AUROC overall", "delta_auroc", "overall", "All stays"),
                                          ("B. \u0394gap, male \u2212 female", "delta_gap", "sex", "Female"),
                                          ("C. \u0394gap, White \u2212 Black", "delta_gap", "race", "Black")]):
    rr = sel(DD, fam, part, lvl)
    for i, r in enumerate(rr):
        a.errorbar(r.estimate, len(rr) - i, xerr=[[r.estimate - r.lo], [r.hi - r.estimate]], fmt="o", ms=4, color="#1f5fa8", capsize=2)
    a.axvline(0, color="#444", lw=0.8, ls="--"); a.set_title(ttl, fontsize=8.5, loc="left")
axs[0].set_yticks(range(1, len(dom_order) + 1)); axs[0].set_yticklabels(dom_order[::-1], fontsize=8)
fig.supxlabel("Effect of swapping ONE SOFA-2 domain into SOFA-1 (as charted); 95% CI", fontsize=8); fig.tight_layout(); fig.savefig("fig/fig6_decomposition.png", dpi=200); plt.close()
print("figures done")
