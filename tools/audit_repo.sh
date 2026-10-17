#!/bin/bash
# Read-only audit of the project repository. Run from ~/mimic-sofa-research :  bash audit_repo.sh
echo "== 1. git log with dates"; git log --format='%h  %ad  %s' --date=iso
echo; echo "== 2. commits cited in the manuscript exist?"
for c in 8709b89 9c79350 3002996 c1f02e2 6cd34e0 e81c720 47bd0f0; do git cat-file -e ${c}^{commit} 2>/dev/null && echo "  $c ok" || echo "  $c MISSING"; done
echo; echo "== 3. what commit 47bd0f0 changed"; git show --stat --format='%h  %ad  %s' --date=iso 47bd0f0 2>/dev/null | head -15
echo; echo "== 4. history of the HEADLINE line in 06 (was it set before any outcome run?)"
git log -L'/^HEADLINE/,+1:src/06_auc_bootstrap.py' --format='%h  %ad  %s' --date=iso -s 2>/dev/null | head -10
echo; echo "== 5. first mortality results file time vs protocol commit"
stat -f "  %Sm  %N" -t "%Y-%m-%d %H:%M:%S" outputs/tables/auc_results_hospital_death_strict.csv outputs/tables/analysis_manifest_hospital_death_strict.json 2>/dev/null
echo "  protocol commit date: $(git log -1 --format=%ad --date=iso 8709b89)"
echo; echo "== 6. manifest commit hashes (commit each run used)"
python3 - << 'PY'
import glob, json, os, subprocess
for f in sorted(glob.glob("outputs/tables/*manifest*.json")):
    c = json.load(open(f)).get("git_commit", "")[:7]
    d = subprocess.run(["git", "log", "-1", "--format=%ad", "--date=iso", c], capture_output=True, text=True).stdout.strip() if c else ""
    print(f"  {os.path.basename(f):75s} {c}  {d}")
PY
echo; echo "== 7. files that must exist"
for f in configs/protocol_locked.yaml configs/race_mapping.csv src/01_cohort.py src/02_extract.py src/03_sofa1.py src/04_sofa2.py src/05_quality_checks.py src/06_auc_bootstrap.py src/08_calibration.py src/09_domain_decomposition.py src/10_spo2_by_group.py src/common.py manuscript/build_final.py manuscript/figs.py manuscript/data.py manuscript/check_manuscript.py tools/amend_protocol.py outputs/tables/cohort_PRIVATE_do_not_share.csv; do [ -e "$f" ] && echo "  ok       $f" || echo "  MISSING  $f"; done
echo "  (sofa_mapping.csv, plausible_ranges_INTERIM.yaml, mimic_itemid_map_v0.2.csv, 07_bootstrap, 09_sensitivity are retired by amendment 2; their absence is expected)"
echo; echo "== 8. tracked files that could hold patient data (should be empty)"
git ls-files | grep -Ei '\.(csv|gz|parquet|zip)$|PRIVATE|physionet' | sed 's/^/  TRACKED  /'
git ls-files -z | xargs -0 ls -l 2>/dev/null | awk '$5 > 1000000 {print "  LARGE   ", $5, $9}'
echo "  .gitignore:"; sed 's/^/    /' .gitignore
echo; echo "== 9. identifier columns or row counts in result files (outputs/tables; PRIVATE files are expected here and are never shared)"
python3 - << 'PY'
import glob, re, pandas as pd
idc = re.compile(r"stay_id|subject_id|hadm_id|charttime|intime|outtime|specimen_id", re.I)
for f in sorted(glob.glob("outputs/tables/*.csv")):
    d = pd.read_csv(f, nrows=2); n = sum(1 for _ in open(f)) - 1
    if "PRIVATE" in f: continue
    if any(idc.search(c) for c in d.columns) or n > 2000: print("  CHECK", f, n)
print("  scan done (no CHECK lines = no identifier columns, no file over 2000 rows)")
PY
