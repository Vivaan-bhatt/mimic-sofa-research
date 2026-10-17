Build the final manuscript from your own result files. Run from the repository root unless stated.
0. pip3 install python-docx matplotlib
1. unzip -o ~/Downloads/manuscript_final.zip          (creates manuscript/ and tools/)
2. bash tools/audit_repo.sh                           (read-only: look for MISSING, TRACKED, CHECK lines)
3. python3 tools/amend_protocol.py                    then   git add configs && git commit -m "protocol amendment 2"
4. open -e manuscript/authors.txt                     (replace the five bracketed lines with your own text; lines left in brackets print as 'to be provided by the authors')
5. cd manuscript && python3 figs.py && python3 build_final.py && python3 check_manuscript.py
   -> SOFA2_vs_SOFA1_MIMIC-IV_manuscript.docx; the last command must print ALL CHECKS PASSED
6. cd .. && git add manuscript tools && git commit -m "manuscript build scripts"
Nothing in the document is typed by hand: numbers come from outputs/tables, age and care units from your private cohort file, commits and dates from git log.
