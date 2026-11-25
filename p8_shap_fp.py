"""Phase 8 – SHAP-based false-positive analysis on VALIDATION predictions of the frozen models.
Outputs, per pipeline:
  p8_cases_<name>.csv          sampled FP / TP / TN (and near-miss) cases with their top-10 SHAP words
  p8_faithfulness_<name>.csv   score drop after deleting the top-5 attributed words
  p8_coding_sheet_<name>.csv   SHUFFLED, BLIND sheet (no case type / score) for manual coding
  p8_coef_check_<name>.csv     (linear only) SHAP vs model-coefficient cross-check"""
import json, re, sys
import numpy as np, pandas as pd, shap
from config import PROC, MODELS, RESULTS, LABEL, SEED
from scoring import DEFAULT_THR, load, score

rng = np.random.default_rng(SEED)

def sample_cases(v, name, thr_sel):
    pred = v.score >= DEFAULT_THR[name]
    v = v.assign(cell=np.select([pred & (v.y == 1), pred & (v.y == 0), ~pred & (v.y == 0)], ["TP", "FP", "TN"], "FN"))
    fp = v[v.cell == "FP"]
    still = int((fp.score >= thr_sel).sum())
    fp_s = fp if len(fp) <= 100 else fp.sample(100, random_state=SEED)
    parts = [fp_s.assign(case="FP")]
    if len(fp) < 30:
        parts.append(v[v.cell == "TN"].nlargest(30, "score").assign(case="near_miss"))
    taken = pd.concat(parts).job_id
    for c in ("TP", "TN"):
        pool = v[(v.cell == c) & ~v.job_id.isin(taken)]
        parts.append(pool.sample(min(50, len(pool)), random_state=SEED).assign(case=c))
    return pd.concat(parts, ignore_index=True), len(fp), still

# ---------- linear pipelines ----------
def explain_linear(name, pipe, cases, train_texts):
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    bg = vec.transform(pd.Series(train_texts).sample(1000, random_state=SEED))
    expl = shap.LinearExplainer(clf, shap.maskers.Independent(bg, max_samples=1000))
    X = vec.transform(cases.text)
    sv = expl(X).values
    vocab = vec.get_feature_names_out()
    top = []
    for i in range(X.shape[0]):
        nz = X[i].nonzero()[1]                                  # only features present in the advert
        order = nz[np.argsort(-sv[i, nz])]
        top.append([(vocab[j], float(sv[i, j])) for j in order[:10] if sv[i, j] > 0])
    # Cross-check with coefficients: top positive coefficients overall
    coef = clf.coef_.ravel()
    pd.DataFrame({"feature": vocab[np.argsort(-coef)[:50]], "coef": np.sort(coef)[::-1][:50]}) \
        .to_csv(RESULTS / f"p8_coef_check_{name}.csv", index=False)
    return top

# ---------- DistilBERT ----------
def merge_subwords(tokens, values):
    words, vals, cur, cv = [], [], "", 0.0
    for t, v in zip(tokens, values):
        cur += t; cv += v
        if t.endswith((" ", "\n")) or not t:
            if cur.strip(): words.append(cur.strip()); vals.append(cv)
            cur, cv = "", 0.0
    if cur.strip(): words.append(cur.strip()); vals.append(cv)
    return words, np.array(vals)

def explain_bert(model, cases):
    import torch
    tok, m = model
    def f(texts):
        enc = tok(list(texts), truncation=True, max_length=512, padding=True, return_tensors="pt").to(m.device)
        with torch.no_grad():
            return torch.softmax(m(**enc).logits, -1)[:, 1].cpu().numpy()
    expl = shap.Explainer(f, shap.maskers.Text(tok), algorithm="partition")
    top = []
    for t in cases.text:
        e = expl([t], max_evals=500, batch_size=16)
        w, v = merge_subwords(e[0].data, e[0].values)
        o = np.argsort(-v)
        top.append([(w[j], float(v[j])) for j in o[:10] if v[j] > 0])
    return top

# ---------- faithfulness: delete top-5 words, record score change ----------
def delete_words(text, words):
    for w in words:
        text = re.sub(rf"(?i)(?<!\w){re.escape(w)}(?!\w)", " ", text)
    return re.sub(r"\s+", " ", text)

def main(names):
    df = pd.read_csv(PROC / "partitions.csv", keep_default_na=False)
    texts = df.set_index("job_id").text
    thr = json.load(open(MODELS / "thresholds.json"))
    for name in names:
        v = pd.read_csv(RESULTS / f"val_scores_{name}.csv")
        cases, n_fp, still = sample_cases(v, name, thr[name]["threshold"])
        cases["text"] = cases.job_id.map(texts)
        model = load(name)
        top = explain_bert(model, cases) if name == "distilbert" else \
              explain_linear(name, model, cases, df[df.split == "train"].text)
        cases["top10"] = [json.dumps(t) for t in top]

        top5 = [[w for w, _ in t[:5]] for t in top]
        reduced = [delete_words(tx, ws) for tx, ws in zip(cases.text, top5)]
        cases["score_after_delete_top5"] = score(name, model, reduced)
        cases["faithfulness_drop"] = cases.score - cases.score_after_delete_top5
        cases.drop(columns="text").to_csv(RESULTS / f"p8_cases_{name}.csv", index=False)
        cases.groupby("case").faithfulness_drop.describe().to_csv(RESULTS / f"p8_faithfulness_{name}.csv")

        sheet = cases.sample(frac=1, random_state=SEED)[["job_id", "text", "top10"]].copy()
        sheet.insert(0, "coding_id", range(1, len(sheet) + 1))
        sheet["categories (fill in, ; separated)"] = ""
        sheet.to_csv(RESULTS / f"p8_coding_sheet_{name}.csv", index=False)
        print(f"{name}: {n_fp} FPs at default threshold, {still} still flagged at selected threshold; "
              f"{len(cases)} cases explained")

if __name__ == "__main__":
    main(sys.argv[1:] or ["logreg", "svm", "distilbert"])
