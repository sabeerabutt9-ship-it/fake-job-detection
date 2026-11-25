"""Phase 7 – single test evaluation of the FROZEN pipelines, default vs selected thresholds,
with 95% group-stratified bootstrap intervals (2,000 resamples) and paired differences.
Run this ONCE, after p6_protocol_record.json has been shared with your supervisor."""
import itertools, json, time
import numpy as np, pandas as pd
from config import PROC, MODELS, RESULTS, LABEL, SEED
from metrics import threshold_metrics, ranking_metrics
from scoring import PIPELINES, DEFAULT_THR, load, score

B = 2000
KEYS = ["precision", "recall", "f1", "fpr", "mcc"]

def group_bootstrap_indices(groups, y, rng):
    """Resample near-duplicate groups (not records) with replacement, stratified by whether a group contains fraud."""
    df = pd.DataFrame({"g": groups, "y": y, "i": np.arange(len(y))})
    members = df.groupby("g").i.apply(np.array)
    strat = df.groupby("g").y.max()
    pools = [strat.index[strat == k].values for k in (0, 1)]
    while True:
        chosen = np.concatenate([rng.choice(p, size=len(p), replace=True) for p in pools])
        yield np.concatenate(members.loc[chosen].values)

def all_metrics(y, s, thr):
    return {**threshold_metrics(y, s, thr), **ranking_metrics(y, s)}

def main():
    te = pd.read_csv(PROC / "partitions.csv", keep_default_na=False).query("split == 'test'").reset_index(drop=True)
    y, g = te[LABEL].values, te.group.values
    thr_sel = json.load(open(MODELS / "thresholds.json"))
    scores, rows = {}, []
    for name in PIPELINES:
        if name not in thr_sel:
            continue
        m = load(name)
        t0 = time.time(); scores[name] = score(name, m, te.text); secs = time.time() - t0
        for tname, t in [("default", DEFAULT_THR[name]), ("selected", thr_sel[name]["threshold"])]:
            rows.append({"pipeline": name, "threshold": tname, "value": t, **all_metrics(y, scores[name], t),
                         "inference_s_per_1000": secs / len(te) * 1000})
    pd.DataFrame(rows).to_csv(RESULTS / "p7_test_point_estimates.csv", index=False)
    pd.DataFrame({"job_id": te.job_id, "y": y, **{f"score_{k}": v for k, v in scores.items()}}) \
        .to_csv(RESULTS / "test_scores.csv", index=False)

    # Bootstrap (paired: every quantity uses the same resample in each iteration)
    rng = np.random.default_rng(SEED); gen = group_bootstrap_indices(g, y, rng); boot = []
    for b in range(B):
        idx = next(gen); yb = y[idx]; rec = {}
        for name, s in scores.items():
            for tname, t in [("default", DEFAULT_THR[name]), ("selected", thr_sel[name]["threshold"])]:
                mm = all_metrics(yb, s[idx], t)
                for k in KEYS + ["roc_auc", "pr_auc"]:
                    rec[f"{name}|{tname}|{k}"] = mm[k]
        boot.append(rec)
    boot = pd.DataFrame(boot)
    ci = boot.quantile([0.025, 0.975]).T.rename(columns={0.025: "lo", 0.975: "hi"})
    ci.to_csv(RESULTS / "p7_bootstrap_ci.csv")

    diffs = []
    names = list(scores)
    for a, b_ in itertools.combinations(names, 2):                          # RQ1: pipeline vs pipeline
        for tname in ("default", "selected"):
            for k in KEYS + ["roc_auc", "pr_auc"]:
                d = boot[f"{a}|{tname}|{k}"] - boot[f"{b_}|{tname}|{k}"]
                diffs.append({"comparison": f"{a} - {b_}", "threshold": tname, "metric": k,
                              "lo": d.quantile(.025), "hi": d.quantile(.975)})
    for name in names:                                                      # RQ3: selected - default
        for k in KEYS:
            d = boot[f"{name}|selected|{k}"] - boot[f"{name}|default|{k}"]
            diffs.append({"comparison": f"{name}: selected - default", "threshold": "-", "metric": k,
                          "lo": d.quantile(.025), "hi": d.quantile(.975)})
    dd = pd.DataFrame(diffs); dd["supported (CI excludes 0)"] = (dd.lo > 0) | (dd.hi < 0)
    dd.to_csv(RESULTS / "p7_paired_differences.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())

if __name__ == "__main__":
    main()
