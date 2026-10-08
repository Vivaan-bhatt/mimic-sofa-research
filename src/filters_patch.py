def patch(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:70], s.count(old))
        s = s.replace(old, new)
    open(path, "w").write(s)

for f in ["src/06_auc_bootstrap.py", "src/08_calibration.py"]:
    patch(f, [
        ('"sofa2": s2[a.sofa2_col]})',
         '"sofa2": s2[a.sofa2_col],\n                       "age": coh["age"], "careunit": coh["first_careunit"].fillna("Unknown")})'),
        ('    ap.add_argument("--selftest", action="store_true")',
         '    ap.add_argument("--age-lt", type=int, default=None, help="keep stays with age below N (prespecified insurance sensitivity)")\n'
         '    ap.add_argument("--careunit", default=None, help="keep stays whose first care unit contains this text, e.g. (MICU)")\n'
         '    ap.add_argument("--selftest", action="store_true")'),
        ('    if a.complete_case:\n',
         '    if a.age_lt is not None:\n        df = df[df["age"] < a.age_lt]\n'
         '    if a.careunit:\n        df = df[df["careunit"].str.contains(a.careunit, case=False, regex=False)]\n'
         '    if a.complete_case:\n'),
        ('+ ("_cc" if a.complete_case else "")',
         '+ (f"_age-lt{a.age_lt}" if a.age_lt is not None else "") + (("_cu-" + "".join(ch for ch in a.careunit if ch.isalnum())) if a.careunit else "") + ("_cc" if a.complete_case else "")'),
    ])
