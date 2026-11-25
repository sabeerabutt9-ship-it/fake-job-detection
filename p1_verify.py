"""Phase 1 – Data acquisition and verification."""
import hashlib, json
import pandas as pd
from config import RAW, RESULTS, LABEL, EXPECTED_ROWS, EXPECTED_FRAUD

sha = hashlib.sha256(RAW.read_bytes()).hexdigest()
df = pd.read_csv(RAW)
n, n_fraud = len(df), int(df[LABEL].sum())
report = {"file": RAW.name, "sha256": sha, "rows": n, "fraudulent": n_fraud,
          "rows_match": n == EXPECTED_ROWS, "fraud_match": n_fraud == EXPECTED_FRAUD}
(RESULTS / "p1_verification.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
assert report["rows_match"] and report["fraud_match"], "Counts differ from published totals – stop and check."
