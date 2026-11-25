"""Phase 3 – Duplicate handling and grouped, stratified 70/15/15 split."""
import numpy as np, pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from config import PROC, RESULTS, LABEL, SEED, NEAR_DUP_SIM

def exact_dedup(df, log):
    keep = []
    for _, g in df.groupby("text", sort=False):
        if g[LABEL].nunique() > 1:                                   # conflicting labels -> remove all
            log += [{"job_id": j, "reason": "exact_dup_label_conflict"} for j in g.job_id]
        else:
            keep.append(g.index[0])
            log += [{"job_id": j, "reason": f"exact_dup_merged_into_{g.job_id.iloc[0]}"} for j in g.job_id.iloc[1:]]
    return df.loc[sorted(keep)].reset_index(drop=True)

def similar_pairs(X, thr, chunk=1000):
    """All pairs (i<j) with cosine >= thr. TF-IDF rows are L2-normalised so dot = cosine."""
    rows, cols, sims = [], [], []
    XT = X.T.tocsc()
    for s in range(0, X.shape[0], chunk):
        S = (X[s:s + chunk] @ XT).toarray()
        i, j = np.nonzero(S >= thr - 1e-9)
        i = i + s
        m = j > i
        rows.append(i[m]); cols.append(j[m]); sims.append(S[i[m] - s, j[m]])
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(sims)

def groups_at(n, r, c, sim, thr):
    m = sim >= thr
    g = coo_matrix((np.ones(m.sum()), (r[m], c[m])), shape=(n, n))
    return connected_components(g, directed=False)

def main():
    df = pd.read_csv(PROC / "clean.csv", keep_default_na=False)
    log = []
    df = exact_dedup(df, log)

    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(df.text)  # text only
    r, c, s = similar_pairs(X, 0.85)                       # compute once at lowest sensitivity level

    # Sensitivity report at 0.85 / 0.90 / 0.95
    sens = {t: groups_at(len(df), r, c, s, t)[0] for t in (0.85, NEAR_DUP_SIM, 0.95)}
    pd.Series(sens, name="n_groups").to_csv(RESULTS / "p3_group_sensitivity.csv")

    # 50 random pairs with similarity 0.85–0.95 for manual inspection
    band = np.where((s >= 0.85) & (s <= 0.95))[0]
    pick = np.random.default_rng(SEED).choice(band, size=min(50, len(band)), replace=False)
    pd.DataFrame({"job_a": df.job_id.values[r[pick]], "job_b": df.job_id.values[c[pick]], "cosine": s[pick],
                  "text_a": df.text.values[r[pick]], "text_b": df.text.values[c[pick]],
                  "is_true_near_duplicate (fill in)": ""}).to_csv(RESULTS / "p3_pairs_to_inspect.csv", index=False)

    _, df["group"] = groups_at(len(df), r, c, s, NEAR_DUP_SIM)
    mixed = df.groupby("group")[LABEL].nunique().gt(1).sum()
    print(f"records={len(df)} groups={df.group.nunique()} mixed-label groups (possible label noise)={mixed}")

    # StratifiedGroupKFold: 20 folds -> 14 train / 3 val / 3 test
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    fold = np.empty(len(df), int)
    for k, (_, idx) in enumerate(sgkf.split(df, df[LABEL], df.group)):
        fold[idx] = k + 1
    df["split"] = np.select([fold <= 14, fold <= 17], ["train", "val"], "test")

    # Check 1: no group in more than one partition
    assert df.groupby("group").split.nunique().max() == 1, "group leakage!"
    # Check 2: max cosine test vs train < threshold
    tr, te = np.where(df.split == "train")[0], np.where(df.split == "test")[0]
    mx = 0.0
    for st in range(0, len(te), 1000):
        mx = max(mx, (X[te[st:st + 1000]] @ X[tr].T).max())
    print("max test-train cosine:", round(float(mx), 4))
    assert mx < NEAR_DUP_SIM, "near-duplicate across test/train!"

    print(df.groupby("split")[LABEL].agg(["size", "sum", "mean"]))
    df[["job_id", "text", LABEL, "group", "split"]].to_csv(PROC / "partitions.csv", index=False)
    pd.DataFrame(log, columns=["job_id", "reason"]).to_csv(RESULTS / "p3_duplicate_log.csv", index=False)

if __name__ == "__main__":
    main()
