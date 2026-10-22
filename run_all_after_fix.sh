set -e
python3 src/06_auc_bootstrap.py
python3 src/06_auc_bootstrap.py --window lookback6
python3 src/06_auc_bootstrap.py --age-lt 65
python3 src/06_auc_bootstrap.py --complete-case
python3 src/06_auc_bootstrap.py --careunit "(CCU)"
python3 src/06_auc_bootstrap.py --careunit "(CVICU)"
python3 src/06_auc_bootstrap.py --careunit "(MICU)"
python3 src/06_auc_bootstrap.py --careunit "(MICU/SICU)"
python3 src/06_auc_bootstrap.py --careunit "(SICU)"
python3 src/06_auc_bootstrap.py --careunit "(TSICU)"
python3 src/06_auc_bootstrap.py --sofa2-tag strict
python3 src/06_auc_bootstrap.py --sofa2-tag strict_sed_broad_ne_half
python3 src/06_auc_bootstrap.py --sofa2-col sofa2_total_norrt_criteria
python3 src/06_auc_bootstrap.py --outcome icu_death_strict
python3 src/08_calibration.py
python3 src/08_calibration.py --age-lt 65
python3 src/08_calibration.py --outcome icu_death_strict
python3 src/09_domain_decomposition.py
python3 src/09_domain_decomposition.py --outcome icu_death_strict
python3 src/10_spo2_by_group.py
