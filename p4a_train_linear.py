"""Phase 4 – TF-IDF + Logistic Regression and TF-IDF + Linear SVM.
Fit on TRAIN only, score on VAL by fraud-class PR-AUC. Test data are never loaded here."""
import itertools, json, time
import joblib, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score
from config import PROC, MODELS, RESULTS, LABEL, SEED, TIE_TOL, set_seed

GRIDS = {
    "logreg": {"C": [0.01, 0.1, 1, 10, 100], "ngram": [(1, 1), (1, 2)], "cw": [None, "balanced"]},
    "svm":    {"C": [0.01, 0.1, 1, 10],      "ngram": [(1, 1), (1, 2)], "cw": [None, "balanced"]},
}

def make(name, C, ngram, cw):
    vec = TfidfVectorizer(ngram_range=ngram, min_df=2, max_df=0.95, sublinear_tf=True,
                          max_features=200_000, norm="l2", lowercase=True)
    clf = (LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000, C=C, class_weight=cw, random_state=SEED)
           if name == "logreg" else
           LinearSVC(penalty="l2", loss="squared_hinge", max_iter=10_000, C=C, class_weight=cw, random_state=SEED))
    return Pipeline([("tfidf", vec), ("clf", clf)])

def score(pipe, texts):
    return pipe.predict_proba(texts)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(texts)

def simplicity(cfg):   # lower = simpler: unigrams, stronger regularisation (smaller C), no class weight
    return (cfg["ngram"][1], cfg["C"], cfg["cw"] is not None)

def main():
    set_seed()
    df = pd.read_csv(PROC / "partitions.csv", keep_default_na=False)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    for name, grid in GRIDS.items():
        rows = []
        for C, ngram, cw in itertools.product(grid["C"], grid["ngram"], grid["cw"]):
            t0 = time.time()
            pipe = make(name, C, ngram, cw).fit(tr.text, tr[LABEL])
            rows.append({"C": C, "ngram": ngram, "cw": cw,
                         "train_pr_auc": average_precision_score(tr[LABEL], score(pipe, tr.text)),
                         "val_pr_auc": average_precision_score(va[LABEL], score(pipe, va.text)),
                         "fit_seconds": time.time() - t0})
            print(name, rows[-1])
        res = pd.DataFrame(rows)
        res["gap"] = res.train_pr_auc - res.val_pr_auc
        res["overfit_flag"] = res.gap > 0.10
        res.to_csv(RESULTS / f"p4_grid_{name}.csv", index=False)

        best = res.val_pr_auc.max()
        cands = [r for r in rows if r["val_pr_auc"] >= best - TIE_TOL]
        chosen = min(cands, key=simplicity)
        t0 = time.time()
        pipe = make(name, chosen["C"], chosen["ngram"], chosen["cw"]).fit(tr.text, tr[LABEL])  # not refit on train+val
        train_time = time.time() - t0
        joblib.dump(pipe, MODELS / f"{name}.joblib")
        pd.DataFrame({"job_id": va.job_id, "group": va.group, "y": va[LABEL],
                      "score": score(pipe, va.text)}).to_csv(RESULTS / f"val_scores_{name}.csv", index=False)
        json.dump({**{k: chosen[k] for k in ("C", "ngram", "cw")}, "val_pr_auc": chosen["val_pr_auc"],
                   "train_seconds": train_time}, open(MODELS / f"{name}_config.json", "w"), indent=2, default=str)
        print("SELECTED", name, chosen)

if __name__ == "__main__":
    main()
