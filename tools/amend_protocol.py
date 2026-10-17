#!/usr/bin/env python3
"""Append protocol amendment 2 to configs/protocol_locked.yaml. Run from the repository root:  python3 tools/amend_protocol.py
Every date, hash and time below is read from git and the file system when the script runs; nothing is typed in."""
import datetime, glob, os, re, subprocess, sys
PATH = sys.argv[1] if len(sys.argv) > 1 else "configs/protocol_locked.yaml"
def git(*a): return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
head = git("rev-parse", "--short", "HEAD"); assert head, "run this inside the repository"
lock = git("log", "-1", "--format=%h %ad", "--date=short", "8709b89") or "commit 8709b89 not found in this repository"
first = git("log", "--reverse", "--format=%h %ad", "--date=short", "-S", 'HEADLINE = "sofa1_ett_charted"', "--", "src/06_auc_bootstrap.py").splitlines()
files = sorted(glob.glob("outputs/tables/auc_results_*.csv"))
earliest = min((os.path.getmtime(f), f) for f in files) if files else None
ts = datetime.datetime.fromtimestamp(earliest[0]).strftime("%Y-%m-%d %H:%M") + " (" + os.path.basename(earliest[1]) + ")" if earliest else "no result files found"
txt = open(PATH).read()
assert "amendments:" in txt, "no amendments block in the protocol"
if "id: 2" in txt: sys.exit("amendment 2 is already in the file; nothing changed")
after = txt.split("amendments:", 1)[1]
assert not re.search(r"^[A-Za-z_]+:", after, re.M), "the amendments block is not the last section of the file; edit by hand"
NOT_DONE = ("a domain-available cohort; first ICU stay per patient; a first-hour plus 24-hour window; SOFA-1 with the SOFA-2 brain rules; "
            "excluding patients with a psychotic or bipolar disorder diagnosis and adding dexmedetomidine to the delirium drugs; counting mixed-specimen blood gases; "
            "counting high-flow nebulizer oxygen as high-flow nasal oxygen; counting paused infusion segments; excluding peritoneal dialysis; "
            "including operating-room and recovery-room urine; restricting to stays of at least 24 hours")
block = f"""  - id: 2
    date: "{datetime.date.today().isoformat()}"
    made_after_primary_results: true
    repository_head_when_recorded: "{head}"
    protocol_lock_commit: "{lock}"
    headline_constant_first_appears_in_commit: "{first[0] if first else 'not found by git log -S'}"
    earliest_auc_result_file_time: "{ts}"
    changes:
      - "Headline comparator: SOFA-1 as charted is the headline comparator (HEADLINE constant in src/06_auc_bootstrap.py, which labels the primary comparisons). The two SOFA-1 versions remain co-primary in the protocol text and both are reported in full."
      - "Prespecified analyses that were NOT performed: {NOT_DONE}."
      - "Superseded references: score thresholds are implemented in src/03_sofa1.py and src/04_sofa2.py and plausible ranges in src/common.py (RANGES); the files configs/sofa_mapping.csv, configs/plausible_ranges_INTERIM.yaml and mimic_itemid_map_v0.2.csv, and the scripts 07_bootstrap and 09_sensitivity, were never created and are not used."
      - "Post hoc: the domain decomposition was extended to separate the SpO2/FiO2 fallback from the new respiratory thresholds after the first decomposition was seen."
"""
if not txt.endswith("\n"): txt += "\n"
open(PATH, "w").write(txt + block)
try:
    import yaml; yaml.safe_load(open(PATH)); print("YAML parses correctly")
except ImportError: print("(PyYAML not installed; YAML not validated)")
print(block)
print("Now commit it:  git add configs && git commit -m 'protocol amendment 2: headline comparator, analyses not performed, superseded file references'")
