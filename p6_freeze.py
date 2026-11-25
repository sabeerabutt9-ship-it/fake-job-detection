"""Phase 6 – protocol freeze. Writes a dated record to share with your supervisor BEFORE testing."""
import datetime, hashlib, json, platform, subprocess
from importlib.metadata import version, PackageNotFoundError
from config import MODELS, RESULTS, ROOT

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def pkg(n):
    try: return version(n)
    except PackageNotFoundError: return None

try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
except Exception:
    commit = "NOT A GIT REPO – commit your code first!"

record = {
    "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "git_commit": commit,
    "python": platform.python_version(),
    "packages": {p: pkg(p) for p in ["numpy", "pandas", "scikit-learn", "torch", "transformers", "shap"]},
    "configs": {n: json.load(open(MODELS / f"{n}_config.json"))
                for n in ["logreg", "svm", "distilbert"] if (MODELS / f"{n}_config.json").exists()},
    "thresholds": json.load(open(MODELS / "thresholds.json")),
    "artefact_sha256": {p.name: sha(p) for p in MODELS.glob("*.joblib")} |
                       ({"distilbert/model.safetensors": sha(MODELS / "distilbert" / "model.safetensors")}
                        if (MODELS / "distilbert" / "model.safetensors").exists() else {}),
    "partitions_sha256": sha(ROOT / "data" / "processed" / "partitions.csv"),
}
(RESULTS / "p6_protocol_record.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
