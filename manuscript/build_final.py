import math, re, os, glob, json, subprocess
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from data import *

# ------------------------------------------------------------------ numbers and claim checks
O = lambda df, comp: get(df, "delta_auroc", "overall", "All stays", comp)
au = lambda df, s, p="overall", l="All stays": get(df, "auroc", p, l, score=s)
dg = lambda df, p, l, c: get(df, "delta_gap", p, l, c)
gp = lambda df, p, l, s: get(df, "gap", p, l, score=s)
excl0 = lambda r: (r.lo > 0) or (r.hi < 0)
CH_, CC_ = "sofa1_ett_charted", "sofa1_community"
cit = lambda r, d=3: f"{sg(r.estimate,d,True)}, 95% CI {sg(r.lo,d,True)} to {sg(r.hi,d,True)}"
N = MAN["n_stays"]; NS = MAN["n_subjects"]; ND = MAN["deaths"]
nd_icu = int(au(I, "sofa2").deaths)
# claims used in the text -------------------------------------------------------
assert all(excl0(O(H, c)) and O(H, c).estimate > 0 for c in COMPS)
assert excl0(dg(H, "sex", "Female", CH_)) and dg(H, "sex", "Female", CH_).estimate > 0
assert excl0(dg(H, "race", "Black", CH_)) and dg(H, "race", "Black", CH_).estimate > 0
assert excl0(dg(H, "insurance", "Private", CH_)) and dg(H, "insurance", "Private", CH_).estimate < 0
others_ch = [dg(H, p, l, CH_) for p, l in [("race", "Hispanic/Latino"), ("race", "Asian"), ("race", "Other"), ("insurance", "Medicaid")]]
assert not any(excl0(r) for r in others_ch)
assert not any(excl0(dg(H, p, l, CC_)) for p, lv in SUBS.items() for l in lv if l != REF[p]), "community-comparator gap changes must all include zero"
assert get(H, "auroc", "race", "Black", score="sofa2").estimate > 0
assert not excl0(get(H, "delta_auroc", "race", "Black", CH_))
dec = lambda dom, fam="delta_auroc", part="overall", lvl="All stays", comp=CH_, df=DD: df[(df.swapped_domain == dom) & (df.family == fam) & (df.partition == part) & (df.level == lvl) & (df.sofa1_comparator == comp)].iloc[0]
resp, pfo, brain = dec("respiratory"), dec("respiratory (P/F only)"), dec("brain")
fallback_share = 100 * (1 - pfo.estimate / resp.estimate)
assert resp.estimate > 0.017 and brain.estimate < resp.estimate / 3 and 60 < fallback_share < 85
cal_ch = [CH[(CH.family == "delta_calib_disparity_logOE") & (CH.comparator == CH_)].estimate.abs().max(), CH[(CH.family == "delta_calib_disparity_logOE") & (CH.comparator == CC_)].estimate.abs().max()]
narrow = pd.read_csv(f"{S}/auc_results_hospital_death_strict_s2-strict.csv")
_cc = pd.read_csv(f"{S}/auc_results_hospital_death_strict_cc.csv"); _a65 = pd.read_csv(f"{S}/auc_results_hospital_death_strict_age-lt65.csv")
assert excl0(dg(I, "sex", "Female", CH_)) and excl0(dg(I, "race", "Black", CH_)) and not excl0(dg(I, "insurance", "Private", CH_)) and excl0(dg(I, "insurance", "Medicaid", CH_))
assert not excl0(dg(_cc, "sex", "Female", CH_)) and not excl0(dg(_cc, "race", "Black", CH_)) and excl0(dg(_cc, "insurance", "Private", CH_)) and dg(_cc, "insurance", "Private", CH_).estimate < 0
assert not excl0(dg(_a65, "race", "Black", CH_)) and excl0(dg(_a65, "sex", "Female", CH_)) and dg(_a65, "sex", "Female", CH_).estimate > 0
sec = lambda f: pd.read_csv(f"{S}/auc_results_hospital_death_lookback6.csv" if f == "lookback6" else f"{S}/auc_results_hospital_death_strict_{f}.csv")
se_w = (SPO2[(SPO2.level == "White")].iloc[0].hi - SPO2[(SPO2.level == "White")].iloc[0].lo) / 3.92
se_b = (SPO2[(SPO2.level == "Black")].iloc[0].hi - SPO2[(SPO2.level == "Black")].iloc[0].lo) / 3.92
sw, sb = SPO2[SPO2.level == "White"].iloc[0], SPO2[SPO2.level == "Black"].iloc[0]
sp_diff = sb.mean_diff - sw.mean_diff; sp_se = math.sqrt(se_w ** 2 + se_b ** 2)

dcomm = lambda dom: DD[(DD.swapped_domain == dom) & (DD.family == "delta_auroc") & (DD.partition == "overall") & (DD.level == "All stays") & (DD.sofa1_comparator == CC_)].iloc[0]
fallback_share_comm = 100 * (1 - dcomm("respiratory (P/F only)").estimate / dcomm("respiratory").estimate)
a65c = pd.read_csv(f"{S}/calib_results_hospital_death_strict_age-lt65.csv")
a65_md = get(a65c, "delta_calib_disparity_logOE", "insurance", "Private", CH_)
def _spread(df, p, lv): v = [get(df, "oe", p, l, score="sofa2").estimate for l in lv]; return max(v) - min(v)
sp_all, sp_65 = _spread(CH, "race", SUBS["race"]), _spread(a65c, "race", SUBS["race"])
assert abs(sp_65 - sp_all) < 0.05 and 85 < fallback_share_comm < 95
FL0 = FLOW.iloc[0]; assert int(FLOW.iloc[4].icu_stays) == N
_ma = [json.load(open(f)) for f in glob.glob(f"{S}/analysis_manifest_*.json")]; _mc = [json.load(open(f)) for f in glob.glob(f"{S}/calibration_manifest_*.json")]
assert len({m["B"] for m in _ma}) == 1 and len({m["B"] for m in _mc}) == 1 and len({m["seed"] for m in _ma + _mc}) == 1, "replicates or seed differ between runs: the text must be changed"
BA, BC, SEED = _ma[0]["B"], _mc[0]["B"], _ma[0]["seed"]
assert BC == 2000, "calibration replicates changed: update the decomposition/SpO2 sentence"
PRIM = MAN["git_commit"][:7]
_n4n = int(DS2N[DS2N.domain == "brain_pts"].pts_4.iloc[0]); _n4b = int(DS2B[DS2B.domain == "brain_pts"].pts_4.iloc[0])
assert os.path.exists(COHORT), "cohort_PRIVATE_do_not_share.csv not found: build inside the repository (it is never shared)"
_coh = pd.read_csv(COHORT); assert len(_coh) == N, (len(_coh), N)
AGE_MED, AGE_Q1, AGE_Q3 = _coh["age"].median(), _coh["age"].quantile(.25), _coh["age"].quantile(.75)
TOK = {"@@BA@@": f"{BA:,}", "@@BC@@": f"{BC:,}", "@@SEED@@": str(SEED), "@@PRIM@@": PRIM, "@@N4NARROW@@": f"{_n4n:,}", "@@N4BROAD@@": f"{_n4b:,}", "@@BRAINPCT@@": f"{100*(1-_n4b/_n4n):.0f}"}
_a = [l.rstrip("\n") for l in open("authors.txt")] if os.path.exists("authors.txt") else []
AUTH = {}
for i, k in enumerate(["authors", "affiliations", "contributions", "funding", "competing"]):
    v = _a[i].strip() if i < len(_a) else ""
    AUTH[k] = "[to be provided by the authors]" if (not v or (v.startswith("[") and v.endswith("]"))) else v
# ------------------------------------------------------------------ docx helpers
doc = Document()
sec0 = doc.sections[0]; sec0.page_width, sec0.page_height = Inches(8.5), Inches(11)
for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"): setattr(sec0, m, Inches(1))
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.space_after = Pt(6); st.paragraph_format.line_spacing = 1.08
for name, size in (("Heading 1", 13), ("Heading 2", 11.5)):
    h = doc.styles[name]; h.font.name = "Times New Roman"; h.font.size = Pt(size); h.font.bold = True; h.font.color.rgb = RGBColor(0, 0, 0)
    rf = h.element.rPr.rFonts
    for att in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"): rf.attrib.pop(qn(att), None)
    for att in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"): rf.set(qn(att), "Times New Roman")
    h.paragraph_format.space_before = Pt(12 if name == "Heading 1" else 8); h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True

_MAP = {1: 2, 2: 3, 3: 1, 4: 4, 5: 5, 6: 6}
def _cm(t): return re.sub(r"\[(\d(?:,\d)*)\]", lambda m: "[" + ",".join(str(x) for x in sorted(_MAP[int(n)] for n in m.group(1).split(","))) + "]", t)
def P(text="", bold_lead=None, italic=False, size=None, align=None, after=None, nocite=False):
    if not nocite: text = _cm(text)
    for _k, _v in TOK.items(): text = text.replace(_k, _v)
    p = doc.add_paragraph()
    if bold_lead:
        r = p.add_run(bold_lead + " "); r.bold = True
        if size: r.font.size = Pt(size)
    r = p.add_run(text); r.italic = italic
    if size: r.font.size = Pt(size)
    if align: p.alignment = align
    if after is not None: p.paragraph_format.space_after = Pt(after)
    return p

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill); tcPr.append(shd)

def TBL(headers, rows, widths, size=8, bold_rows=(), group_rows=()):
    t = doc.add_table(rows=1, cols=len(headers)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    def fill(cell, text, bold=False, w=None, fillc=None, align=None):
        cell.width = Inches(w); cell.text = ""
        p = cell.paragraphs[0]; p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = 1.0
        r = p.add_run(str(text)); r.font.size = Pt(size); r.bold = bold
        if align: p.alignment = align
        if fillc: shade(cell, fillc)
    for i, h in enumerate(headers):
        fill(t.rows[0].cells[i], h, True, widths[i], "D9E2F3", WD_ALIGN_PARAGRAPH.CENTER if i else None)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        grp = ri in group_rows
        for i, v in enumerate(row):
            fill(cells[i], v, bold=(ri in bold_rows or grp), w=widths[i], fillc="F2F2F2" if grp else None, align=WD_ALIGN_PARAGRAPH.CENTER if i and not grp else None)
    tr = t.rows[0]._tr.get_or_add_trPr(); th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); tr.append(th)
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true"); trPr.append(cs)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t

def CAP(label, text):
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(3); p.paragraph_format.keep_with_next = True
    r = p.add_run(label + " "); r.bold = True; r.font.size = Pt(9.5)
    r = p.add_run(text); r.font.size = Pt(9.5)

def FIG(path, label, text, width=6.3):
    doc.add_picture(path, width=Inches(width)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(8)
    r = p.add_run(label + " "); r.bold = True; r.font.size = Pt(9.5); r = p.add_run(text); r.font.size = Pt(9.5)

def footer_page_numbers():
    p = sec0.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Page "); r.font.size = Pt(9)
    for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        run = p.add_run(); run.font.size = Pt(9)
        if t:
            fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), t); run._r.append(fc)
        else:
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = txt; run._r.append(it)
footer_page_numbers()
pct = lambda a, b: f"{100 * a / b:.1f}%"

# ------------------------------------------------------------------ title + abstract
t = P("Does Replacing SOFA-1 with SOFA-2 Change Mortality Discrimination and Calibration Across Patient Subgroups? A Paired Analysis in MIMIC-IV", align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
for r in t.runs: r.bold = True; r.font.size = Pt(15)
P(AUTH["authors"], align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
P(AUTH["affiliations"], italic=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, after=10)

a_h, a_c = O(H, CH_), O(H, CC_)
P(f"The Sequential Organ Failure Assessment (SOFA) score describes organ dysfunction in intensive care. SOFA-2, published in 2025, updates the 1996 score (SOFA-1). It is unknown whether the update changes mortality discrimination and calibration within patient subgroups.", "Background.")
P(f"We analyzed {N:,} first ICU stays per hospitalization ({NS:,} adults) in MIMIC-IV v3.1, scoring SOFA-1 (two prespecified conventions for intubated patients) and SOFA-2 from the same data over the same first 24 hours. The primary outcome was in-hospital mortality; ICU mortality was secondary. We estimated AUROC overall and by sex, race and insurance, the paired change between scores, subgroup gaps in AUROC and the change in those gaps, and descriptive calibration, using a paired patient-level bootstrap ({BA:,} replicates). The protocol was committed before any analysis relating scores to mortality.", "Methods.")
P(f"{ND:,} patients ({pct(ND, N)}) died in hospital. AUROC was {au(H,'sofa2').estimate:.3f} for SOFA-2, {au(H,CH_).estimate:.3f} for SOFA-1 as charted and {au(H,CC_).estimate:.3f} for SOFA-1 under the community convention; SOFA-2 minus SOFA-1 was {sg(a_h.estimate,3,True)} ({sg(a_h.lo,3,True)} to {sg(a_h.hi,3,True)}) and {sg(a_c.estimate,3,True)} ({sg(a_c.lo,3,True)} to {sg(a_c.hi,3,True)}). Swapping single domains showed that the respiratory domain accounted for the whole gain, about {fallback_share:.0f}% of it through the SpO2/FiO2 fallback. Subgroup gaps changed against SOFA-1 as charted (male\u2013female {sg(dg(H,'sex','Female',CH_).estimate,3,True)}; White\u2013Black {sg(dg(H,'race','Black',CH_).estimate,3,True)}), driven by the brain-domain rule, but not against the community convention; the White\u2013Black change was not seen in patients under 65, and neither change was consistent across ICU types. Calibration differences between groups were large but changed little between scores (|\u0394 log O/E| \u2264 {max(cal_ch):.3f}); insurance-group differences largely disappeared in patients under 65, whereas race-group differences did not.", "Results.")
P("SOFA-2 modestly improved overall discrimination for in-hospital mortality, mainly through its respiratory domain. Changes in subgroup performance gaps were small, depended on how SOFA-1 handles intubated patients, partly reflected age, and were not consistent across ICU types. These are descriptive associations in a single database; we draw no causal or fairness conclusions.", "Conclusions.")
P("SOFA-2; SOFA; organ dysfunction; critical care; MIMIC-IV; subgroup analysis; discrimination; calibration", "Keywords:", size=10)

# ------------------------------------------------------------------ introduction
doc.add_heading("1. Introduction", 1)
P("The Sequential Organ Failure Assessment (SOFA) score summarizes dysfunction in six organ systems on a 0\u201324 scale and is among the most widely used severity measures in intensive care [3]. It was designed to describe organ dysfunction, not to predict death, yet SOFA values are routinely related to mortality. In 2025 an international group published SOFA-2, which keeps six organ systems but adds contemporary organ-support treatments and revised thresholds; its authors reported good predictive validity for ICU mortality across ten cohorts [1,2].")
P("A score that discriminates well overall can still discriminate worse within a subgroup, and a change to the score can alter that pattern. Because SOFA-2 changes how respiratory, circulatory, renal and neurological dysfunction are scored, including how intubated and sedated patients are handled, it is reasonable to ask whether it changes performance differently across patient groups. To our knowledge this has not been examined within a single cohort using paired comparisons.")
P("We therefore compared SOFA-1 and SOFA-2 on the same patients, outcome and observation window in MIMIC-IV. We asked (i) whether SOFA-2 changes overall mortality discrimination, (ii) whether it changes discrimination and calibration within subgroups defined by sex, race and insurance, (iii) whether it changes the gaps between subgroups, and (iv) which domain changes drive any difference. Both scores were developed to describe organ dysfunction, and we use mortality only as an external check on validity; we do not frame either score as a prediction model.")

# ------------------------------------------------------------------ methods
doc.add_heading("2. Methods", 1)
doc.add_heading("2.1 Data and cohort", 2)
P(f"We used MIMIC-IV v3.1 [4], a de-identified database of admissions to a tertiary care center. We included adult ICU stays and kept the first ICU stay of each hospitalization, which mirrors the exclusion of ICU readmissions within the same hospital stay in the SOFA-2 derivation [1]. Of {FL0.icu_stays:,} ICU stays ({FL0.hospitalizations:,} hospitalizations, {FL0.patients:,} patients), {int(FLOW.iloc[3].icu_stays) - N:,} stays with missing or invalid ICU timestamps were excluded, leaving {N:,} stays from {NS:,} patients (Fig. 1). Patients could contribute more than one hospitalization; all resampling was done by patient.")
doc.add_heading("2.2 Observation window, missing data and outcomes", 2)
P("Both scores used the worst value of each component in the first 24 hours after ICU admission (from admission to 24 hours, half-open), as in the SOFA-2 derivation. A component with no measurement was scored 0, the primary missing-data rule recommended for SOFA-2 [1]; complete-case and a window extended 6 hours before admission were prespecified sensitivity analyses. The primary outcome was in-hospital mortality. ICU mortality (death during the ICU stay) was secondary because it was the outcome in the SOFA-2 validation [1].")
doc.add_heading("2.3 Score implementation", 2)
P("Each score was implemented independently from raw tables on a shared cleaned input set, so differences arise only from the score definitions. Physiologic plausibility limits were set by us (the consensus ranges in the SOFA-2 supplement were not available to us) and are listed in the repository.")
P("SOFA-1 followed the original definition [3]. Because the original paper does not say how to score the verbal GCS component in intubated patients, we prespecified two conventions and report both throughout: (a) the MIMIC community convention, which scores intubated patients as GCS 15; and (b) as charted, where the verbal component of an intubated patient is scored 1. The as-charted version is the prespecified headline comparator. Respiratory support meant invasive ventilation; PaO2/FiO2 used arterial samples only. Missing GCS components were carried forward for up to 6 hours and otherwise set to normal; urine output was used only for stays of at least 22 hours (scaled to 24 hours); cardiovascular points used dopamine, dobutamine, epinephrine and norepinephrine only, as in the original definition; infusion rows with a paused status were ignored.", "SOFA-1.")
P("SOFA-2 followed Table 2 of the SOFA-2 manuscript [1], including the SpO2/FiO2 fallback (used only when SpO2 was below 98%, a charted FiO2 was available from the preceding 4 hours, and no arterial PaO2/FiO2 was recorded within 60 minutes), ventilatory support defined as invasive ventilation, non-invasive ventilation or high-flow nasal oxygen with a maximum of 2 respiratory points without such support, ECMO, vasopressor scoring from the summed norepinephrine and epinephrine dose and other agents for infusions lasting at least 1 hour, urine output per kilogram per hour, renal replacement therapy including chronic dialysis and the criteria-based rule, and one brain point for treatment with antipsychotics for delirium. For sedated patients the last GCS before sedation was used (0 points if none) and, for patients whose GCS could not be fully assessed, the motor component alone, using our own mapping of the Table 2 descriptors to points (motor 6 = 0 points, 5 = 1, 4 = 2, 3 = 3, 1–2 = 4). Because Table 2 does not define sedation, we prespecified a broad definition (continuous infusion of a sedative or fentanyl; a bolus of midazolam, lorazepam, fentanyl or ketamine in the previous 2 hours (propofol and dexmedetomidine boluses were not captured); or a Richmond Agitation\u2013Sedation Scale score of \u22123 or below in the previous 4 hours). The infusion-only definition was a sensitivity analysis. Ceiling-of-treatment rules and the base-versus-salt form of norepinephrine could not be determined in MIMIC-IV. Peritoneal dialysis counted as renal replacement therapy, and chronic dialysis was identified from diagnosis codes (end-stage renal disease or dialysis dependence) or the charted dialysis flag.", "SOFA-2.")
doc.add_heading("2.4 Subgroups", 2)
P("Subgroups were fixed from category counts alone, before any outcome was examined: sex (male, female); race and ethnicity grouped as White, Black, Hispanic/Latino, Asian and Other, with unknown, unable-to-obtain and declined records described but excluded from race comparisons; and insurance (Medicare, Medicaid, Private; other or missing described only). Reference groups were male, White and Medicare, the largest group in each factor. Race and insurance are recorded administrative categories; insurance is a rough socioeconomic proxy and is strongly tied to age in this cohort.")
doc.add_heading("2.5 Statistical analysis", 2)
P("Discrimination was the area under the receiver operating characteristic curve (AUROC). The paired difference in AUROC (SOFA-2 minus SOFA-1) was estimated with a patient-level cluster bootstrap (@@BA@@ replicates, seed @@SEED@@; the calibration analyses used @@BC@@ replicates, and the decomposition and SpO2 analyses used 2,000 replicates, with the same seed and resampling) that resamples patients and keeps each stay\u2019s SOFA-1 and SOFA-2 values together; we report percentile 95% intervals and two-sided bootstrap p-values and, as a cross-check for the overall difference, the paired DeLong test [5]. We never inferred a difference from the overlap of separate intervals. A subgroup gap was the AUROC of the reference group minus the AUROC of the comparison group; the change in the gap was the SOFA-2 gap minus the SOFA-1 gap, with intervals from the same replicates. Because the sign of a gap depends on which group is the reference, we also report the gap itself.")
P("SOFA is not a probability model, so calibration analysis was descriptive. We report observed mortality by score band, a pooled logistic curve (logit mortality = a + b \u00d7 score, fitted on all stays) as a reference, the ratio of observed to expected deaths (O/E) in each subgroup relative to that curve, and calibration disparity, defined as log(O/E of the group) minus log(O/E of the reference group), and its change between scores. No model was tuned to improve calibration. Primary comparisons were the overall and subgroup AUROC changes and the changes in gaps for the headline comparator and in-hospital mortality; all other results are labeled co-primary or exploratory, and we emphasize effect sizes with intervals over p-values.")
doc.add_heading("2.6 Sensitivity, exploratory and post hoc analyses", 2)
P("Prespecified sensitivity analyses were: ICU mortality; infusion-only sedation; a 6-hour pre-admission lookback window; norepinephrine dose halved; SOFA-2 without the criteria-based renal replacement rule; complete-case stays; patients under 65 years (because insurance is tied to age); and analyses within each major ICU type. Two further analyses were prespecified as exploratory: a domain decomposition, in which one SOFA-2 domain at a time replaces its SOFA-1 counterpart in the SOFA-1 total, and a comparison of respiratory points derived from paired SpO2 and arterial PaO2 measurements (within 10 minutes, same FiO2) by group. Separating the SpO2/FiO2 fallback from the new respiratory thresholds within the decomposition was added after seeing the first decomposition and is post hoc. Prespecified analyses that were not performed: a domain-available cohort; first ICU stay per patient; a first-hour plus 24-hour window; SOFA-1 with the SOFA-2 brain rules; excluding patients with a psychotic or bipolar disorder diagnosis and adding dexmedetomidine to the delirium drugs; counting mixed-specimen blood gases; counting high-flow nebulizer oxygen as high-flow nasal oxygen; counting paused infusion segments; excluding peritoneal dialysis; including operating-room and recovery-room urine; and restricting to stays of at least 24 hours.")
doc.add_heading("2.7 Protocol and reproducibility", 2)
P("The analysis protocol was committed to version control (commit 8709b89) before any analysis relating scores to mortality; overall mortality counts and death timing had been viewed during cohort construction and informed the ICU-mortality definition. One amendment was made before outcomes were examined (commit 3002996): an outcome-blind audit showed that the number of stays with a SOFA-2 brain score of 4 fell by @@BRAINPCT@@% (@@N4NARROW@@ to @@N4BROAD@@) when sedation was defined broadly, so the broad definition became primary and the infusion-only definition a sensitivity analysis. The primary results were generated at commit @@PRIM@@ (recorded in every result manifest). A second amendment, recorded after the primary results, designates SOFA-1 as charted as the headline comparator in the analysis script (the locked protocol lists both SOFA-1 versions as co-primary, and both are reported in full), lists the prespecified analyses that were not performed, and retires references in the protocol to files that were never created. The age, care-unit, domain-decomposition and SpO2 analyses were listed in the locked protocol; separating the SpO2 fallback within the decomposition is post hoc. All commits are listed in Supplementary Table S7, and every analysis run is listed with its replicates, seed and commit in Supplementary Table S8. Patient-level data are not shared.")

# ------------------------------------------------------------------ results
doc.add_heading("3. Results", 1)
doc.add_heading("3.1 Cohort and score distributions", 2)
fem = au(H, "sofa2", "sex", "Female")
P(f"The cohort comprised {N:,} stays (Fig. 1; Table 1). Median age was {AGE_MED:.0f} years (interquartile range {AGE_Q1:.0f}\u2013{AGE_Q3:.0f}), {pct(fem.n, N)} were women, {ND:,} ({pct(ND, N)}) died in hospital and {nd_icu:,} ({pct(nd_icu, N)}) died in the ICU. Median total score in the first 24 hours was {D[CC_]['median']} (IQR {D[CC_]['q1']}\u2013{D[CC_]['q3']}) for SOFA-1 under the community convention, {D[CH_]['median']} ({D[CH_]['q1']}\u2013{D[CH_]['q3']}) for SOFA-1 as charted and {D['sofa2']['median']} ({D['sofa2']['q1']}\u2013{D['sofa2']['q3']}) for SOFA-2 (means {D[CC_]['mean']:.2f}, {D[CH_]['mean']:.2f} and {D['sofa2']['mean']:.2f}; Fig. 2). Measurement coverage differed by domain: a respiratory score could be computed in {obs(DS1,'resp_pts'):.1f}% of stays for SOFA-1 (arterial PaO2/FiO2 only) and {obs(DS2B,'resp_pts'):.1f}% for SOFA-2 (with the SpO2/FiO2 fallback), and bilirubin was available in {obs(DS1,'liver_pts'):.1f}% of stays.")
doc.add_heading("3.2 Overall discrimination", 2)
ai, ci_ = O(I, CH_), O(I, CC_)
P(f"For in-hospital mortality, AUROC was {sg(au(H,'sofa2').estimate)} (95% CI {sg(au(H,'sofa2').lo)} to {sg(au(H,'sofa2').hi)}) for SOFA-2, {sg(au(H,CH_).estimate)} ({sg(au(H,CH_).lo)} to {sg(au(H,CH_).hi)}) for SOFA-1 as charted and {sg(au(H,CC_).estimate)} ({sg(au(H,CC_).lo)} to {sg(au(H,CC_).hi)}) for SOFA-1 under the community convention (Table 2). The paired difference was {ci(a_h,3,True)} against SOFA-1 as charted and {ci(a_c,3,True)} against the community convention; the DeLong cross-check agreed. For ICU mortality, AUROC was {sg(au(I,'sofa2').estimate)}, {sg(au(I,CH_).estimate)} and {sg(au(I,CC_).estimate)}, with differences of {ci(ai,3,True)} and {ci(ci_,3,True)}, similar in size to the gain reported in the SOFA-2 derivation [1].")
doc.add_heading("3.3 Subgroup discrimination and gaps", 2)
f_, m_ = get(H, "delta_auroc", "sex", "Female", CH_), get(H, "delta_auroc", "sex", "Male", CH_)
b_ = get(H, "delta_auroc", "race", "Black", CH_); w_ = get(H, "delta_auroc", "race", "White", CH_)
P(f"SOFA-2 AUROC was similar across sex ({sg(au(H,'sofa2','sex','Male').estimate)} male, {sg(au(H,'sofa2','sex','Female').estimate)} female) and race ({sg(au(H,'sofa2','race','Asian').estimate)}\u2013{sg(au(H,'sofa2','race','Hispanic/Latino').estimate)}) and was lower for Medicare ({sg(au(H,'sofa2','insurance','Medicare').estimate)}) than Medicaid ({sg(au(H,'sofa2','insurance','Medicaid').estimate)}) or Private ({sg(au(H,'sofa2','insurance','Private').estimate)}) coverage (Table 3, Fig. 3). Against SOFA-1 as charted, the gain was larger in men ({cit(m_)}) than women ({cit(f_)}) and in White ({cit(w_)}) than Black patients ({cit(b_)}).")
sx, wb, mp = dg(H, "sex", "Female", CH_), dg(H, "race", "Black", CH_), dg(H, "insurance", "Private", CH_)
g1x, g2x = gp(H, "sex", "Female", CH_).estimate, gp(H, "sex", "Female", "sofa2").estimate
g1b, g2b = gp(H, "race", "Black", CH_).estimate, gp(H, "race", "Black", "sofa2").estimate
assert g1x < 0 and g2x < 0 and abs(g2x) < abs(g1x) and g1b < 0 and abs(g2b) < abs(g1b)
P(f"With SOFA-1 as charted, men and Black patients had lower AUROC than women and White patients (gaps of {sg(g1x)} and {sg(g1b)}), so a positive change means the gap moved toward zero. Under SOFA-2 the gaps were {sg(g2x)} and {sg(g2b)}; the changes were {ci(sx,3,True)} for male\u2013female and {ci(wb,3,True)} for White\u2013Black. The Medicare\u2013Private gap widened slightly ({cit(mp)}). Other changes had intervals that included zero (Table 4, Fig. 4). Against the community convention, no change in any gap was distinguishable from zero. In ICU mortality the sex and White\u2013Black changes were similar ({cit(dg(I,"sex","Female",CH_))}; {cit(dg(I,"race","Black",CH_))}); the Medicare\u2013Private change was not distinguishable from zero, and the Medicare\u2013Medicaid gap changed slightly ({cit(dg(I,"insurance","Medicaid",CH_))}; Supplementary Table S6).")
doc.add_heading("3.4 Calibration", 2)
oe = lambda p, l, s: get(CH, "oe", p, l, score=s)
P(f"At the same score, observed mortality differed substantially between groups (Table 5, Fig. 5): O/E for SOFA-2 was {sg(oe('sex','Male','sofa2').estimate,2)} in men and {sg(oe('sex','Female','sofa2').estimate,2)} in women; {sg(oe('insurance','Medicare','sofa2').estimate,2)}, {sg(oe('insurance','Medicaid','sofa2').estimate,2)} and {sg(oe('insurance','Private','sofa2').estimate,2)} for Medicare, Medicaid and Private; and {sg(oe('race','Hispanic/Latino','sofa2').estimate,2)} to {sg(oe('race','Asian','sofa2').estimate,2)} across race groups. In the full cohort these disparities changed little between scores: the largest absolute change in calibration disparity was {cal_ch[0]:.3f} log units against SOFA-1 as charted and {cal_ch[1]:.3f} against the community convention (about {100*(math.exp(cal_ch[0])-1):.0f}% and {100*(math.exp(cal_ch[1])-1):.0f}% in relative O/E). Observed mortality at mid-range scores was visibly lower for SOFA-1 as charted than for the other two scores (Fig. 5A). In patients under 65 the insurance differences narrowed (O/E {sg(get(pd.read_csv(f'{S}/calib_results_hospital_death_strict_age-lt65.csv'),'oe','insurance','Medicare',score='sofa2').estimate,2)} Medicare, {sg(get(pd.read_csv(f'{S}/calib_results_hospital_death_strict_age-lt65.csv'),'oe','insurance','Private',score='sofa2').estimate,2)} Private), consistent with an age effect for insurance; the spread of O/E across race groups did not narrow ({sp_all:.2f} in the full cohort, {sp_65:.2f} under 65), and the Medicare\u2013Private calibration disparity changed more against SOFA-1 as charted in this subset ({cit(a65_md)}).")
doc.add_heading("3.5 Which SOFA-2 changes drive the results", 2)
bn = dec("brain", "delta_gap", "sex", "Female"); bw = dec("brain", "delta_gap", "race", "Black"); bm = dec("brain", "delta_gap", "insurance", "Private")
cv, kd = dec("cardiovascular"), dec("kidney")
P(f"Replacing only the respiratory domain reproduced the entire gain in AUROC ({cit(resp)}, against SOFA-1 as charted; full SOFA-2 {sg(a_h.estimate,3,True)}); the brain domain alone contributed {ci(brain,3,True)}, and the cardiovascular ({sg(cv.estimate,3,True)}) and kidney ({sg(kd.estimate,3,True)}) changes slightly lowered discrimination (Fig. 6, Supplementary Table S3). Restricting the respiratory change to the new PaO2/FiO2 thresholds and support rules gave only {ci(pfo,3,True)}, so about {fallback_share:.0f}% of the respiratory gain ({fallback_share_comm:.0f}% against the community convention) came from scoring additional patients through the SpO2/FiO2 fallback (post hoc split). The changes in subgroup gaps came from the brain domain instead: swapping it alone changed the male\u2013female gap by {ci(bn,3,True)} and the White\u2013Black gap by {ci(bw,3,True)}, and the Medicare\u2013Private gap by {ci(bm,3,True)}.")
doc.add_heading("3.6 Sensitivity analyses", 2)
s_n = get(narrow, "delta_auroc", "overall", "All stays", CH_); a_n = au(narrow, "sofa2")
P(f"Results were robust to a 6-hour pre-admission window, halving the norepinephrine dose, and removing the criteria-based renal rule (overall AUROC change {sg(O(sec('lookback6'),CH_).estimate,3,True)}, {sg(O(sec('s2-strict_sed_broad_ne_half'),CH_).estimate,3,True)} and {sg(O(sec('sofa2_total_norrt_criteria'),CH_).estimate,3,True)}; gap changes essentially unchanged; Supplementary Table S1). The result was sensitive to how sedation is defined: with the infusion-only definition SOFA-2 reached an AUROC of {sg(a_n.estimate)} and the change against SOFA-1 as charted was {ci(s_n,3,True)}, because deep sedation or unresponsiveness is itself prognostic. In the {sec('cc').pipe(lambda d: int(get(d,'auroc','overall','All stays',score='sofa2').n)):,} complete-case stays the AUROC change was smaller ({cit(O(sec('cc'),CH_))}); the sex and White\u2013Black gap changes were not distinguishable from zero, and the Medicare\u2013Private gap widened slightly, as in the full cohort ({cit(dg(sec('cc'),'insurance','Private',CH_))}). In patients under 65 the White\u2013Black gap change was no longer distinguishable from zero ({cit(dg(sec('age-lt65'),'race','Black',CH_))}), while the male\u2013female change persisted ({cit(dg(sec('age-lt65'),'sex','Female',CH_))}).")
P("The advantage of SOFA-2 over SOFA-1 as charted was not uniform across ICU types (Supplementary Table S2): it was large in the cardiac-surgery ICU and present in the medical/surgical ICU, and absent or slightly negative in the medical, surgical, coronary care and trauma ICUs. Within the cardiac-surgery ICU, SOFA-1 as charted had the lowest AUROC of the three scores.")
doc.add_heading("3.7 SpO2/FiO2 versus PaO2/FiO2 scoring by group (exploratory)", 2)
P(f"Among {int(SPO2[(SPO2.partition=='race')].n_pairs.sum()):,} paired measurements, the SpO2/FiO2 route gave more respiratory points than the PaO2/FiO2 route on average (Supplementary Table S4); the level of this difference is not interpretable as bias, because pairs were restricted to SpO2 below 98%. The difference between groups was informative: the excess was smaller in Black patients (mean {sb.mean_diff:.2f} points, 95% CI {sb.lo:.2f} to {sb.hi:.2f}) than in White patients ({sw.mean_diff:.2f}, {sw.lo:.2f} to {sw.hi:.2f}), a difference of about {sg(sp_diff,2)} points (approximate 95% CI {sg(sp_diff-1.96*sp_se,2)} to {sg(sp_diff+1.96*sp_se,2)}, normal approximation), with the SpO2 route scoring lower than the arterial route in {sb.pct_sf_lower:.1f}% of Black and {sw.pct_sf_lower:.1f}% of White pairs. The direction is consistent with reports that pulse oximetry can overestimate oxygenation in patients with darker skin [6], but we could not measure arterial saturation and treat this as hypothesis-generating. The share of stays whose respiratory score relied on SpO2 alone was similar across race groups ({SPO2[SPO2.level=='Black'].iloc[0].pct_resp_from_spo2_only:.1f}% Black, {SPO2[SPO2.level=='White'].iloc[0].pct_resp_from_spo2_only:.1f}% White).")

# ------------------------------------------------------------------ discussion
doc.add_heading("4. Discussion", 1)
P(f"In {N:,} first ICU stays, SOFA-2 discriminated in-hospital mortality better than SOFA-1 by {a_h.estimate:.2f} to {a_c.estimate:.2f} AUROC, similar in size to the gain reported in the SOFA-2 derivation for ICU mortality [1]. Three findings qualify that headline. First, the gain came almost entirely from the respiratory domain, and mostly from the SpO2/FiO2 fallback, which scores patients who lack an arterial blood gas; it reflects broader measurement coverage more than new thresholds. Second, the apparent narrowing of the sex and race gaps against SOFA-1 as charted came from the brain-domain rule for intubated and sedated patients. It was absent against the community convention, in the complete-case subset and within most ICU types, and in patients under 65 only the sex change persisted, so it should be read as a feature of one comparison, not a general property of SOFA-2. Third, differences in calibration between groups were large but almost identical for the two scores, and the insurance differences largely disappeared in younger patients, which points to factors outside SOFA, chiefly age, rather than to the score update.")
P("The sedation result deserves emphasis. The footnote of Table 2 that replaces the GCS of a sedated patient with the last pre-sedation value is intended to keep the score about organ dysfunction. When we implemented this broadly, mortality discrimination fell compared with leaving deeply sedated patients scored as unresponsive, because depth of sedation tracks illness severity. Mortality AUROC therefore rewards the opposite of the stated intent of the rule, a reminder that agreement with mortality is not the same as validity for describing organ dysfunction. We chose the broad definition before examining any outcome and report the alternative in full.")
P("The exploratory SpO2 analysis raises a question that needs data we did not extract. If pulse-oximetry error differs by race, scoring respiratory dysfunction from SpO2 could introduce group-dependent measurement differences into a score whose respiratory domain now carries most of its added discriminative value. The paired comparison suggests that the SpO2 route scores relatively fewer points in Black patients; confirming this would require arterial saturation and a formal test of occult hypoxemia.")
doc.add_heading("4.1 Limitations", 2)
for lead, txt in [
    ("Single database and design.", "All data come from one tertiary center and the study is retrospective and observational; results may not transfer to other settings, and we make no causal claims."),
    ("Score implementation.", "SOFA-2 was implemented from Table 2 of the author-accepted manuscript; the published version and the supplement (plausible ranges, mapping tables) were not available to us, so physiologic limits are our own. Ceiling-of-treatment rules, norepinephrine base-versus-salt form and the indication for antipsychotics could not be determined from MIMIC-IV, and the sedation definition and its 2-hour and 4-hour windows are our operationalization. The conclusions depend on the SOFA-1 convention for intubated patients and on the sedation definition, as shown."),
    ("Missing data.", f"Missing components were scored 0 as recommended, but measurement was uneven (a respiratory score could be computed in {obs(DS1,'resp_pts'):.0f}% of stays under SOFA-1 and bilirubin was available in {obs(DS1,'liver_pts'):.0f}%) and differed by group. Complete-case analyses, which are a selected, sicker subset, gave smaller effects with wide intervals."),
    ("Outcome and purpose.", "Both scores describe organ dysfunction; mortality is an indirect check, ICU deaths were defined as death during the ICU stay, and no inference about bedside decisions can be drawn."),
    ("Subgroups.", "Race and insurance are administrative categories; insurance is confounded with age; groups other than White had few deaths, so their intervals are wide, and results for the smallest groups and ICU strata are unstable. Many comparisons were made; only the prespecified primary set should be read as confirmatory."),
    ("Respiratory analysis.", "The SpO2 comparison used paired measurements restricted to SpO2 below 98% and could not assess arterial saturation, so its absolute level is not interpretable and its group difference is exploratory.")]:
    P(txt, lead)
doc.add_heading("5. Conclusion", 1)
P("SOFA-2 modestly improved overall discrimination of in-hospital mortality in MIMIC-IV, chiefly through its respiratory domain. Differences in subgroup performance gaps were small, depended on how SOFA-1 scores intubated patients, partly reflected age, and were not consistent across ICU types, while calibration differences between groups were large but nearly unchanged by the update. These findings argue for transparent reporting of how each score is implemented and for evaluating any use of SOFA-2 beyond describing organ dysfunction within the populations where it will be used.")
doc.add_heading("Declarations", 2)
P("Data are from MIMIC-IV, a de-identified database accessed under the PhysioNet credentialed data use agreement; patient-level data are not redistributed.", size=10)
P(AUTH["contributions"], "Author contributions:", size=10)
P(AUTH["funding"], "Funding:", size=10)
P(AUTH["competing"], "Competing interests:", size=10)
doc.add_heading("References", 2)
refs = ["Vincent JL, Moreno R, Takala J, et al. The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure. Intensive Care Med. 1996;22(7):707-710. doi:10.1007/BF01709751",
        "Ranzani OT, Singer M, Salluh JIF, et al. Development and validation of the Sequential Organ Failure Assessment (SOFA)-2 score. JAMA. 2025;334(23):2090-2103. doi:10.1001/jama.2025.20516",
        "Moreno R, Rhodes A, Ranzani O, et al. Rationale and methodological approach underlying the development of the Sequential Organ Failure Assessment (SOFA)-2 score: a consensus statement. JAMA Netw Open. 2025;8(10):e2545040. doi:10.1001/jamanetworkopen.2025.45040",
        "Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data. 2023;10(1):1. doi:10.1038/s41597-022-01899-x",
        "DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics. 1988;44(3):837-845.",
        "Sjoding MW, Dickson RP, Iwashyna TJ, Gay SE, Valley TS. Racial bias in pulse oximetry measurement. N Engl J Med. 2020;383(25):2477-2478. doi:10.1056/NEJMc2029240"]
for i, r in enumerate(refs, 1): P(f"[{i}] {r}", size=9.5, after=2, nocite=True)

# ------------------------------------------------------------------ main tables / figures
doc.add_page_break()
FIG("fig/fig1_flow.png", "Figure 1.", "Cohort flow.", 5.2)
CAP("Table 1.", "Cohort characteristics (first ICU stay per hospitalization).")
def r1(label, n, d): return [label, f"{n:,} ({pct(n, N)})", f"{d:,} ({pct(d, n)})" if d is not None else "\u2014"]
cu = {"Medical ICU": ("MICU", 18657), "Medical/Surgical ICU": ("MICUSICU", 14087), "Cardiac Vascular ICU": ("CVICU", 13199), "Surgical ICU": ("SICU", 11561), "Coronary Care Unit": ("CCU", 9804), "Trauma SICU": ("TSICU", 9306)}
rows = [["All stays", f"{N:,}", f"{ND:,} ({pct(ND, N)})"], ["Age, median (IQR)", f"{AGE_MED:.0f} ({AGE_Q1:.0f}\u2013{AGE_Q3:.0f})", "\u2014"]]; grp = [0]
rows.append(["Sex"] + ["", ""]); grp.append(len(rows) - 1)
for l in ["Male", "Female"]:
    r = au(H, "sofa2", "sex", l); rows.append(r1("   " + l, int(r.n), int(r.deaths)))
rows.append(["Race / ethnicity"] + ["", ""]); grp.append(len(rows) - 1)
for l in ["White", "Black", "Hispanic/Latino", "Asian", "Other", "Unknown/Declined"]:
    r = au(H, "sofa2", "race", l); rows.append(r1("   " + l, int(r.n), int(r.deaths)))
rows.append(["Insurance"] + ["", ""]); grp.append(len(rows) - 1)
for l in ["Medicare", "Medicaid", "Private", "Other/Missing"]:
    r = au(H, "sofa2", "insurance", l); rows.append(r1("   " + l, int(r.n), int(r.deaths)))
rows.append(["First ICU care unit (largest six)"] + ["", ""]); grp.append(len(rows) - 1)
for lab, (key, nn) in cu.items():
    d = au(sec(f"cu-{key}"), "sofa2"); rows.append(r1("   " + lab, int(d.n), int(d.deaths)))
TBL(["Characteristic", "Stays, n (%)", "In-hospital deaths, n (% of group)"], rows, [3.0, 1.6, 1.9], size=8.5, group_rows=grp)

CAP("Table 2.", "Overall discrimination. AUROC (95% CI); paired differences (SOFA-2 minus SOFA-1) from the patient-level bootstrap; DeLong p-value in the last column.")
rows = []
for lab, df in [("In-hospital mortality (primary)", H), ("ICU mortality (secondary)", I)]:
    dl = get(df, "delta_auroc_delong", "overall", "All stays", CH_)
    rows.append([lab, ci(au(df, CC_)), ci(au(df, CH_)), ci(au(df, "sofa2")), ci(O(df, CC_), 3, True), ci(O(df, CH_), 3, True), pf(dl.p_boot)])
TBL(["Outcome", "SOFA-1 community", "SOFA-1 as charted", "SOFA-2", "\u0394 vs community", "\u0394 vs as charted (headline)", "DeLong p (headline)"], rows, [1.05, 0.95, 0.95, 0.95, 0.95, 0.95, 0.5], size=7.5)

FIG("fig/fig2_distributions.png", "Figure 2.", "Distribution of total scores in the first 24 hours (missing components scored 0).", 6.3)
CAP("Table 3.", "Subgroup AUROC for in-hospital mortality and paired change (SOFA-2 minus SOFA-1). Reference groups are marked with an asterisk; Unknown/Declined race and Other/Missing insurance are not shown.")
rows = []; grp = []
for p, lv in SUBS.items():
    rows.append([p.capitalize()] + [""] * 6); grp.append(len(rows) - 1)
    for l in lv:
        r = au(H, "sofa2", p, l)
        rows.append([f"   {l}{'*' if l == REF[p] else ''}", f"{int(r.n):,} ({int(r.deaths):,})", sg(au(H, CC_, p, l).estimate), sg(au(H, CH_, p, l).estimate), sg(r.estimate), ci(get(H, "delta_auroc", p, l, CC_), 3, True), ci(get(H, "delta_auroc", p, l, CH_), 3, True)])
TBL(["Group", "n (deaths)", "SOFA-1 community", "SOFA-1 as charted", "SOFA-2", "\u0394 vs community (95% CI)", "\u0394 vs as charted (95% CI)"], rows, [1.15, 0.95, 0.7, 0.7, 0.6, 1.3, 1.3], size=7.5, group_rows=grp)
FIG("fig/fig3_delta_auroc.png", "Figure 3.", "Paired change in AUROC for in-hospital mortality, SOFA-2 minus SOFA-1, overall and by subgroup. Bars are 95% patient-level bootstrap intervals.", 5.6)

CAP("Table 4.", "Subgroup AUROC gaps (reference minus group) and their change from SOFA-1 to SOFA-2, in-hospital mortality. A negative gap means the group had higher AUROC than the reference. \u0394gap is the SOFA-2 gap minus the SOFA-1 gap; change in magnitude is |gap SOFA-2| minus |gap SOFA-1| using point estimates.")
rows = []
for p, lv in SUBS.items():
    for l in lv:
        if l == REF[p]: continue
        gC, gH, g2 = gp(H, p, l, CC_), gp(H, p, l, CH_), gp(H, p, l, "sofa2")
        rows.append([f"{REF[p]} \u2212 {l}", sg(gC.estimate, 3, True), sg(gH.estimate, 3, True), sg(g2.estimate, 3, True), ci(dg(H, p, l, CC_), 3, True), ci(dg(H, p, l, CH_), 3, True),
                     f"{sg(abs(g2.estimate) - abs(gH.estimate), 3, True)} / {sg(abs(g2.estimate) - abs(gC.estimate), 3, True)}"])
TBL(["Contrast", "Gap, SOFA-1 community", "Gap, SOFA-1 as charted", "Gap, SOFA-2", "\u0394gap vs community (95% CI)", "\u0394gap vs as charted (95% CI)", "Change in |gap|: as charted / community"], rows, [1.2, 0.65, 0.65, 0.6, 1.2, 1.2, 0.95], size=7.5)
FIG("fig/fig4_delta_gap.png", "Figure 4.", "Change in subgroup AUROC gaps from SOFA-1 to SOFA-2. A positive change means the gap (reference minus group) increased; for most contrasts the SOFA-1 gap was negative, so a positive change reflects a narrower gap (Table 4).", 5.8)

CAP("Table 5.", "Calibration: observed-to-expected deaths (O/E) relative to the pooled score curve, and change in calibration disparity (log O/E of the group minus log O/E of the reference), SOFA-2 minus SOFA-1. In-hospital mortality.")
rows = []; grp = []
for p, lv in SUBS.items():
    rows.append([p.capitalize()] + [""] * 5); grp.append(len(rows) - 1)
    for l in lv:
        r = [f"   {l}{'*' if l == REF[p] else ''}"] + [ci(get(CH, "oe", p, l, score=s), 2) for s in [CC_, CH_, "sofa2"]]
        if l == REF[p]: r += ["reference", "reference"]
        else: r += [ci(get(CH, "delta_calib_disparity_logOE", p, l, CC_), 3, True), ci(get(CH, "delta_calib_disparity_logOE", p, l, CH_), 3, True)]
        r.append("")
        rows.append(r[:6])
TBL(["Group", "O/E SOFA-1 community", "O/E SOFA-1 as charted", "O/E SOFA-2", "\u0394 disparity vs community", "\u0394 disparity vs as charted"], rows, [1.0, 1.1, 1.1, 1.1, 1.1, 1.1], size=7.5, group_rows=grp)
FIG("fig/fig5_calibration.png", "Figure 5.", "Observed in-hospital mortality by score band (Wilson 95% intervals). A: overall, three scores. B: SOFA-2 by insurance group.", 6.3)
FIG("fig/fig6_decomposition.png", "Figure 6.", "Domain decomposition: effect of replacing one SOFA-1 (as charted) domain with its SOFA-2 version. The respiratory (P/F only) row excludes the SpO2/FiO2 fallback (post hoc).", 6.3)

# ------------------------------------------------------------------ supplementary
doc.add_page_break()
doc.add_heading("Supplementary material", 1)
CAP("Supplementary Table S1.", "Sensitivity analyses, in-hospital mortality. \u0394 = SOFA-2 minus SOFA-1 as charted (headline comparator); gap changes are against the same comparator.")
variants = [("Headline (broad sedation, strict window)", H), ("Infusion-only sedation", narrow), ("6-hour pre-admission lookback", sec("lookback6")), ("Norepinephrine dose halved", sec("s2-strict_sed_broad_ne_half")),
            ("Without criteria-based RRT rule", sec("sofa2_total_norrt_criteria")), ("Complete-case stays", sec("cc")), ("Age under 65", sec("age-lt65"))]
rows = []
for lab, df in variants:
    o = au(df, "sofa2")
    rows.append([lab, f"{int(o.n):,} ({int(o.deaths):,})", sg(o.estimate), ci(O(df, CH_), 3, True), sg(dg(df, "sex", "Female", CH_).estimate, 3, True), sg(dg(df, "race", "Black", CH_).estimate, 3, True), sg(dg(df, "insurance", "Private", CH_).estimate, 3, True)])
TBL(["Analysis", "n (deaths)", "SOFA-2 AUROC", "\u0394AUROC (95% CI)", "\u0394gap M\u2212F", "\u0394gap White\u2212Black", "\u0394gap Medicare\u2212Private"], rows, [1.65, 0.9, 0.65, 1.3, 0.6, 0.75, 0.95], size=7.5)
CAP("Supplementary Table S2.", "Analyses within first care unit, in-hospital mortality. AUROC for SOFA-1 community / SOFA-1 as charted / SOFA-2; \u0394 = SOFA-2 minus SOFA-1 as charted; gap changes against SOFA-1 as charted.")
rows = []
for lab, (key, nn) in cu.items():
    df = sec(f"cu-{key}"); o = au(df, "sofa2")
    rows.append([lab, f"{int(o.n):,} ({int(o.deaths):,})", f"{sg(au(df, CC_).estimate)} / {sg(au(df, CH_).estimate)} / {sg(o.estimate)}", ci(O(df, CH_), 3, True), ci(dg(df, "sex", "Female", CH_), 3, True), ci(dg(df, "race", "Black", CH_), 3, True)])
TBL(["Care unit", "n (deaths)", "AUROC (3 scores)", "\u0394AUROC (95% CI)", "\u0394gap male\u2212female", "\u0394gap White\u2212Black"], rows, [1.45, 0.9, 1.35, 1.1, 1.15, 1.15], size=7.5)
CAP("Supplementary Table S3.", "Domain decomposition (swap one SOFA-2 domain into SOFA-1). Comparator: SOFA-1 as charted (upper) and community (lower); in-hospital mortality.")
rows = []; grp = []
for comp, nm in [(CH_, "Comparator: SOFA-1 as charted"), (CC_, "Comparator: SOFA-1 community")]:
    rows.append([nm] + [""] * 4); grp.append(len(rows) - 1)
    for d in ["brain", "respiratory", "respiratory (P/F only)", "cardiovascular", "liver", "kidney", "hemostasis"]:
        rows.append(["   " + d, ci(dec(d, comp=comp), 3, True), ci(dec(d, "delta_gap", "sex", "Female", comp), 3, True), ci(dec(d, "delta_gap", "race", "Black", comp), 3, True), ci(dec(d, "delta_gap", "insurance", "Private", comp), 3, True)])
TBL(["Domain swapped", "\u0394AUROC overall", "\u0394gap male\u2212female", "\u0394gap White\u2212Black", "\u0394gap Medicare\u2212Private"], rows, [1.7, 1.25, 1.25, 1.25, 1.25], size=7.5, group_rows=grp)
CAP("Supplementary Table S4.", "Paired respiratory scoring: SpO2/FiO2-derived minus PaO2/FiO2-derived SOFA-2 respiratory points (pairs within 10 minutes, SpO2 below 98%, same FiO2), by group. Exploratory. Mean difference with patient-level bootstrap 95% CI.")
rows = []
for r in SPO2.itertuples():
    rows.append([f"{r.partition.capitalize()}: {r.level}", f"{int(r.n_stays_in_group):,}", f"{r.pct_resp_from_spo2_only:.1f}%", f"{int(r.n_stays_paired):,}", f"{int(r.n_pairs):,}", f"{r.mean_diff:.2f} ({r.lo:.2f} to {r.hi:.2f})", f"{r.pct_sf_lower:.1f}% / {r.pct_sf_higher:.1f}%"])
TBL(["Group", "Stays", "Respiratory score from SpO2 only", "Stays with pairs", "Pairs", "Mean difference (95% CI)", "SpO2 route lower / higher"], rows, [1.8, 0.65, 0.85, 0.65, 0.5, 1.3, 0.9], size=7.5)
CAP("Supplementary Table S5.", "Percent of stays with each domain observed, by subgroup (SOFA-1 / SOFA-2); respiratory (resp), bilirubin (liver) and all six domains (complete case). Observed percentages do not depend on the sedation rule or on the SOFA-1 version.")
rows = []
for r in MISS.itertuples():
    rows.append([f"{r.group.capitalize()}: {r.level}", f"{int(r.n):,}", f"{r.s1_resp:.1f} / {r.s2_resp:.1f}", f"{r.s1_liver:.1f} / {r.s2_liver:.1f}", f"{r.s1_complete:.1f} / {r.s2_complete:.1f}"])
TBL(["Group", "n", "Resp. observed %", "Liver observed %", "Complete case %"], rows, [1.9, 0.8, 1.3, 1.3, 1.3], size=7.5)
CAP("Supplementary Table S6.", "Subgroup AUROC and change in gaps for ICU mortality (secondary outcome), SOFA-2 minus SOFA-1 as charted (headline) and community.")
rows = []
for p, lv in SUBS.items():
    for l in lv:
        r = au(I, "sofa2", p, l)
        row = [f"{p.capitalize()}: {l}", f"{int(r.n):,} ({int(r.deaths):,})", sg(r.estimate), ci(get(I, "delta_auroc", p, l, CH_), 3, True), ci(get(I, "delta_auroc", p, l, CC_), 3, True)]
        row += ["reference", "reference"] if l == REF[p] else [ci(dg(I, p, l, CH_), 3, True), ci(dg(I, p, l, CC_), 3, True)]
        rows.append(row)
TBL(["Group", "n (deaths)", "SOFA-2 AUROC", "\u0394AUROC vs as charted", "\u0394AUROC vs community", "\u0394gap vs as charted", "\u0394gap vs community"], rows, [1.4, 0.9, 0.55, 1.15, 1.15, 1.15, 1.15], size=7)

# ------------------------------------------------------------------ audit trail (read from the repository, never typed)
doc.add_page_break()
CAP("Supplementary Table S7.", "Analysis audit trail: every commit in the project repository (hash, date, subject), read from git log when this document was built.")
_log = subprocess.run(["git", "-C", "..", "log", "--reverse", "--format=%h|%ad|%s", "--date=short"], capture_output=True, text=True)
assert _log.returncode == 0 and _log.stdout.strip(), "git log failed: build this document inside the repository (manuscript/ folder)"
TBL(["Commit", "Date", "Subject"], [l.split("|", 2) for l in _log.stdout.strip().splitlines()], [0.8, 0.9, 4.9], size=7.5)
CAP("Supplementary Table S8.", "Analysis runs with their recorded settings (from the manifest written by each run). The domain-decomposition and SpO2 analyses write no manifest and used the code default of 2,000 replicates with seed 20251007.")
_rows = []
for f in sorted(glob.glob(f"{S}/*manifest*.json")):
    m = json.load(open(f)); _rows.append([os.path.basename(f).replace("_manifest", "").replace(".json", ""), str(m.get("outcome", "")), str(m.get("window", "")), str(m.get("B", "")), str(m.get("seed", "")), str(m.get("git_commit", ""))[:7], f"{int(m.get('n_stays', 0)):,}"])
TBL(["Run", "Outcome", "Window", "Replicates", "Seed", "Commit", "Stays"], _rows, [3.0, 1.0, 0.6, 0.7, 0.8, 0.6, 0.6], size=7)

doc.save("SOFA2_vs_SOFA1_MIMIC-IV_manuscript.docx")
print("saved")
