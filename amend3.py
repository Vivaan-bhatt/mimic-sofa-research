import datetime, re, subprocess, sys
PATH = "configs/protocol_locked.yaml"
def git(*a): return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
fix = git("log", "-1", "--format=%h %ad", "--date=short", "--grep=fix SOFA-2 kidney")
assert fix, "fix commit not found"
txt = open(PATH).read()
assert "amendments:" in txt and "id: 2" in txt
if "id: 3" in txt: sys.exit("amendment 3 is already in the file; nothing changed")
assert not re.search(r"^[A-Za-z_]+:", txt.split("amendments:", 1)[1], re.M), "amendments block is not last; edit by hand"
block = f"""  - id: 3
    date: "{datetime.date.today().isoformat()}"
    made_after_primary_results: true
    code_commit_with_fix: "{fix}"
    changes:
      - "Post-analysis comparison of src/04_sofa2.py with Table 2 and its footnotes in the published SOFA-2 paper (JAMA 2025;334:2090-2103) found one difference: anuria (0 mL) for at least 12 hours was scored 4 kidney points, whereas the published table scores it 3. Corrected; SOFA-2 scores and all downstream analyses were rerun."
      - "Effect: 4 of 85,228 stays changed from 4 to 3 kidney points. Headline results unchanged at the reported precision (SOFA-2 AUROC 0.762; difference +0.019 against SOFA-1 as charted and +0.027 against community); a few reported values changed in the third decimal (Tables 3, 5, S1, S6 and one Results sentence); no conclusion changed."
      - "Disclosed, not implemented or not verifiable: the footnote g rule ignoring transient (under 1 hour) changes in the oxygenation ratio; long-term home ventilation; ceiling-of-treatment rules. The published supplement (plausible ranges, mapping tables) was not used."
      - "Interpretation choices: mechanical circulatory support scores 4 cardiovascular points on its own (footnote i); VA and VAV ECMO serve as the proxy for ECMO used for cardiovascular indications."
"""
if not txt.endswith("\n"): txt += "\n"
open(PATH, "w").write(txt + block)
try:
    import yaml; yaml.safe_load(open(PATH)); print("YAML parses correctly")
except ImportError: print("(PyYAML not installed; YAML not validated)")
print(block)
