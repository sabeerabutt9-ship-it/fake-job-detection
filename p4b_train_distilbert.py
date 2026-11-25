"""Phase 4 – fine-tuned DistilBERT (run on Google Colab with a GPU).
6 configs: lr {2e-5, 3e-5, 5e-5} x loss {unweighted, inverse-frequency weighted}.
AdamW, wd 0.01, batch 16, 10% warm-up, <=4 epochs, early stopping (patience 1) on val PR-AUC,
fp16, grad clipping 1.0, max 512 tokens with head truncation."""
import json, shutil, time
import numpy as np, pandas as pd, torch
from datasets import Dataset
from sklearn.metrics import average_precision_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer,
                          TrainingArguments, EarlyStoppingCallback)
from config import PROC, MODELS, RESULTS, LABEL, SEED, TIE_TOL, set_seed

CKPT = "distilbert-base-uncased"
LRS = [2e-5, 3e-5, 5e-5]
LOSSES = ["unweighted", "weighted"]

tok = AutoTokenizer.from_pretrained(CKPT)
tok.truncation_side = "right"          # head truncation: keep the first 512 tokens

def encode(df):
    ds = Dataset.from_pandas(df[["text", LABEL]].rename(columns={LABEL: "labels"}).reset_index(drop=True))
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=512), batched=True, remove_columns=["text"])

class WeightedTrainer(Trainer):
    def __init__(self, *a, class_weights=None, **k):
        super().__init__(*a, **k); self.cw = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels"); out = model(**inputs)
        w = None if self.cw is None else self.cw.to(out.logits.device)
        loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=w)
        return (loss, out) if return_outputs else loss

def fraud_prob(logits):
    return torch.softmax(torch.tensor(logits), -1)[:, 1].numpy()

def compute_metrics(p):
    return {"pr_auc": average_precision_score(p.label_ids, fraud_prob(p.predictions))}

def stability(seeds=(43, 44)):
    """Reliability check: retrain the SELECTED config with seeds 43 and 44; report val PR-AUC spread.
    The seed-42 model stays the frozen one."""
    cfg = json.load(open(MODELS / "distilbert_config.json"))
    df = pd.read_csv(PROC / "partitions.csv", keep_default_na=False)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    dtr, dva = encode(tr), encode(va)
    n0, n1 = (tr[LABEL] == 0).sum(), (tr[LABEL] == 1).sum()
    inv = torch.tensor([len(tr) / (2 * n0), len(tr) / (2 * n1)], dtype=torch.float)
    rows = [{"seed": SEED, "val_pr_auc": cfg["val_pr_auc"]}]
    for sd in seeds:
        set_seed(sd)
        model = AutoModelForSequenceClassification.from_pretrained(CKPT, num_labels=2, dropout=0.1)
        args = TrainingArguments(
            output_dir=f"/tmp/stab{sd}", learning_rate=float(cfg["lr"]), weight_decay=0.01,
            per_device_train_batch_size=16, per_device_eval_batch_size=64, num_train_epochs=4,
            warmup_ratio=0.1, max_grad_norm=1.0, fp16=torch.cuda.is_available(), eval_strategy="epoch",
            save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="pr_auc",
            save_total_limit=1, seed=sd, report_to="none")
        t = WeightedTrainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva, processing_class=tok,
                            compute_metrics=compute_metrics,
                            class_weights=inv if cfg["loss"] == "weighted" else None,
                            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)])
        t.train(); rows.append({"seed": sd, "val_pr_auc": t.evaluate()["eval_pr_auc"]})
    pd.DataFrame(rows).to_csv(RESULTS / "p4_distilbert_seed_stability.csv", index=False)
    print(pd.DataFrame(rows))

def main():
    set_seed()
    df = pd.read_csv(PROC / "partitions.csv", keep_default_na=False)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    dtr, dva = encode(tr), encode(va)
    n0, n1 = (tr[LABEL] == 0).sum(), (tr[LABEL] == 1).sum()
    inv = torch.tensor([len(tr) / (2 * n0), len(tr) / (2 * n1)], dtype=torch.float)  # from TRAIN only

    rows = []
    for lr in LRS:
        for loss in LOSSES:
            tag = f"lr{lr}_{loss}"; out = MODELS / "distilbert_runs" / tag
            model = AutoModelForSequenceClassification.from_pretrained(CKPT, num_labels=2, dropout=0.1)
            args = TrainingArguments(
                output_dir=str(out), learning_rate=lr, weight_decay=0.01,
                per_device_train_batch_size=16, per_device_eval_batch_size=64,
                num_train_epochs=4, warmup_ratio=0.1, max_grad_norm=1.0, fp16=torch.cuda.is_available(),
                eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                metric_for_best_model="pr_auc", greater_is_better=True, save_total_limit=1,
                seed=SEED, logging_strategy="epoch", report_to="none")
            trainer = WeightedTrainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva,
                                      processing_class=tok, compute_metrics=compute_metrics,
                                      class_weights=inv if loss == "weighted" else None,
                                      callbacks=[EarlyStoppingCallback(early_stopping_patience=1)])
            t0 = time.time(); trainer.train(); secs = time.time() - t0
            val_pr = trainer.evaluate()["eval_pr_auc"]
            tr_pr = average_precision_score(tr[LABEL], fraud_prob(trainer.predict(dtr).predictions))
            pd.DataFrame(trainer.state.log_history).to_csv(RESULTS / f"p4_curve_distilbert_{tag}.csv", index=False)
            trainer.save_model(str(out / "best")); tok.save_pretrained(str(out / "best"))
            rows.append({"lr": lr, "loss": loss, "train_pr_auc": tr_pr, "val_pr_auc": val_pr,
                         "gap": tr_pr - val_pr, "train_seconds": secs, "path": str(out / "best")})
            print(rows[-1])

    res = pd.DataFrame(rows); res["overfit_flag"] = res.gap > 0.10
    res.to_csv(RESULTS / "p4_grid_distilbert.csv", index=False)
    best = res.val_pr_auc.max()
    # Simpler = unweighted loss, then lower learning rate
    chosen = res[res.val_pr_auc >= best - TIE_TOL].sort_values(["loss", "lr"], key=lambda s: s.map(
        lambda v: 0 if v == "unweighted" else 1 if v == "weighted" else v)).iloc[0]
    final = MODELS / "distilbert"
    shutil.copytree(chosen.path, final, dirs_exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(final)
    pred = Trainer(model=model, args=TrainingArguments(output_dir="/tmp/x", per_device_eval_batch_size=64,
                   report_to="none")).predict(dva)
    pd.DataFrame({"job_id": va.job_id, "group": va.group, "y": va[LABEL],
                  "score": fraud_prob(pred.predictions)}).to_csv(RESULTS / "val_scores_distilbert.csv", index=False)
    json.dump(chosen.drop("path").to_dict(), open(MODELS / "distilbert_config.json", "w"), indent=2, default=str)
    print("SELECTED", chosen.to_dict())

if __name__ == "__main__":
    import sys
    stability() if "--stability" in sys.argv else main()
