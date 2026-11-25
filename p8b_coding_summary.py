"""Phase 8 step 6 + reliability: after you have coded the sheets by hand.
Prevalence of each category among FP / TP / TN with 95% Wilson intervals, and
Cohen's kappa per category on the 20% re-coded subset (re-code >= 2 weeks later).
Usage: python p8b_coding_summary.py logreg
Re-code file expected at results/p8_recode_<name>.csv with columns coding_id, categories."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
from config import RESULTS, SEED
from metrics import wilson

name = sys.argv[1]
sheet = pd.read_csv(RESULTS / f"p8_coding_sheet_{name}.csv", keep_default_na=False)
cases = pd.read_csv(RESULTS / f"p8_cases_{name}.csv")[["job_id", "case"]]
sheet = sheet.merge(cases, on="job_id")                  # un-blind only now
col = "categories (fill in, ; separated)"
sets = sheet[col].map(lambda s: {c.strip() for c in s.split(";") if c.strip()})
cats = sorted(set().union(*sets))

rows = []
for case in sorted(sheet.case.unique()):
    m = sheet.case == case; n = int(m.sum())
    for c in cats:
        k = int(sets[m].map(lambda s: c in s).sum()); lo, hi = wilson(k, n)
        rows.append({"case": case, "category": c, "k": k, "n": n, "prevalence": k / n, "lo": lo, "hi": hi})
pd.DataFrame(rows).to_csv(RESULTS / f"p8_prevalence_{name}.csv", index=False)

# Draw the 20% re-code subset (run once, before re-coding)
sub = RESULTS / f"p8_recode_ids_{name}.csv"
if not sub.exists():
    sheet.sample(frac=0.2, random_state=SEED)[["coding_id", "text", "top10"]].assign(categories="") \
        .to_csv(sub, index=False)
    print("Re-code subset written to", sub, "- code it in >= 2 weeks, save as p8_recode_<name>.csv")
rec = RESULTS / f"p8_recode_{name}.csv"
if rec.exists():
    r = pd.read_csv(rec, keep_default_na=False).merge(sheet[["coding_id", col]], on="coding_id")
    a = r[col].map(lambda s: {c.strip() for c in s.split(";") if c.strip()})
    b = r["categories"].map(lambda s: {c.strip() for c in s.split(";") if c.strip()})
    kap = [{"category": c, "kappa": cohen_kappa_score(a.map(lambda s: c in s), b.map(lambda s: c in s))} for c in cats]
    kd = pd.DataFrame(kap); kd["unreliable (<0.60)"] = kd.kappa < 0.60
    kd.to_csv(RESULTS / f"p8_kappa_{name}.csv", index=False); print(kd)
print(pd.DataFrame(rows).round(3).to_string())
