# Fake Job Advertisement Detection – implementation (MAC capstone)

Each script = one phase of Chapter 3 (Research Procedure). Run from the `src/` folder, in order.

| Step | Script | Report phase | Where to run | Main outputs (results/) |
|---|---|---|---|---|
| 1 | `p1_verify.py` | Phase 1 – acquisition & verification | laptop | `p1_verification.json` (SHA-256, 17,880 / 866) |
| 2 | `p2_clean.py` | Phase 2 – cleaning & input construction | laptop | `clean.csv`, exclusion log, missingness by class |
| 3 | `p3_dedup_split.py` | Phase 3 – duplicates & grouped split | laptop | `partitions.csv`, sensitivity 0.85/0.90/0.95, 50 pairs to inspect |
| 4a | `p4a_train_linear.py` | Phase 4 – LR & SVM grids (20 + 16) | laptop | grid tables, `val_scores_*.csv`, frozen models |
| 4b | `p4b_train_distilbert.py` | Phase 4 – DistilBERT grid (6) | **Colab GPU** | grid table, learning curves, `val_scores_distilbert.csv` |
| 4c | `p4b_train_distilbert.py --stability` | Reliability – seeds 43, 44 | Colab GPU | `p4_distilbert_seed_stability.csv` |
| 5 | `p5_threshold.py` | Phase 5 – threshold (recall ≥ 0.80, min FPR) | laptop | `models/thresholds.json`, FPR at 0.70–0.90 |
| 6 | `p6_freeze.py` | Phase 6 – protocol freeze | laptop | `p6_protocol_record.json` → **send to supervisor** |
| 7 | `p7_test_eval.py` | Phase 7 – single test evaluation | Colab (needs DistilBERT) | point estimates, bootstrap CIs, paired differences |
| 8 | `p8_shap_fp.py` | Phase 8 – SHAP FP analysis | Colab | cases + top-10 words, faithfulness, blind coding sheets |
| 9 | `p8b_coding_summary.py <name>` | Phase 8 step 6 + kappa | laptop | prevalence with Wilson CIs, kappa per category |
| 10 | `streamlit run app/streamlit_app.py` | Demonstration prototype | laptop | interactive demo |

Setup: `pip install -r requirements.txt`; put Kaggle's `fake_job_postings.csv` (EMSCAD) in `data/raw/`.

Rules that protect validity (do not break them):
* Test rows are only read in `p7_test_eval.py`, and only after step 6.
* Nothing is fitted on validation or test data (TF-IDF, class weights, models).
* Selected models are **not** refitted on train+val.
* Commit to Git before running `p6_freeze.py` so the record holds the commit hash.
