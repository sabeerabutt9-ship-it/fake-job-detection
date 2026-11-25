"""Phase 5 – decision-threshold selection on VALIDATION scores only.
Rule: recall >= 0.80 -> lowest FPR -> ties by higher recall; place at midpoint to next lower score."""
import json, numpy as np, pandas as pd
from config import MODELS, RESULTS, RECALL_TARGET
from scoring import PIPELINES

def curve(y, s):
    u = np.unique(s)[::-1]                         # distinct scores, high -> low
    P, N = y.sum(), (1 - y).sum()
    tp = np.array([(y[s >= t]).sum() for t in u]); fp = np.array([((1 - y)[s >= t]).sum() for t in u])
    return u, tp / P, fp / N

def select(y, s, target):
    u, rec, fpr = curve(y, s)
    ok = np.where(rec >= target)[0]
    best = ok[np.lexsort((-rec[ok], fpr[ok]))[0]]       # lowest FPR, then highest recall
    lower = u[best + 1] if best + 1 < len(u) else u[best] - 1e-6
    return (u[best] + lower) / 2, rec[best], fpr[best]

def main():
    out, sens = {}, []
    for name in PIPELINES:
        f = RESULTS / f"val_scores_{name}.csv"
        if not f.exists():
            print("skip", name); continue
        v = pd.read_csv(f); y, s = v.y.values, v.score.values
        thr, r, fp = select(y, s, RECALL_TARGET)
        out[name] = {"threshold": float(thr), "val_recall": float(r), "val_fpr": float(fp)}
        for t in (0.70, 0.75, RECALL_TARGET, 0.85, 0.90):
            _, r2, f2 = select(y, s, t)
            sens.append({"pipeline": name, "recall_level": t, "val_recall": r2, "val_fpr": f2})
    json.dump(out, open(MODELS / "thresholds.json", "w"), indent=2)
    pd.DataFrame(sens).to_csv(RESULTS / "p5_fpr_at_recall_levels.csv", index=False)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
