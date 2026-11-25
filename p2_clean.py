"""Phase 2 – Cleaning and preparation (steps 1–6 of the report)."""
import html, re, unicodedata
import pandas as pd
from config import RAW, PROC, RESULTS, TEXT_FIELDS, LABEL, MIN_CHARS

TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"(#URL_[0-9a-f]+#|https?://\S+|www\.\S+)", re.I)
EMAIL = re.compile(r"(#EMAIL_[0-9a-f]+#|[\w.+-]+@[\w-]+\.[\w.]+)", re.I)
PHONE = re.compile(r"(#PHONE_[0-9a-f]+#|\+?\d[\d\s().-]{7,}\d)", re.I)
WS = re.compile(r"\s+")

def normalise(t) -> str:
    if pd.isna(t):
        return ""                                     # step 3: missing -> empty string
    t = html.unescape(TAG.sub(" ", str(t)))
    t = URL.sub(" [URL] ", t); t = EMAIL.sub(" [EMAIL] ", t); t = PHONE.sub(" [PHONE] ", t)
    t = unicodedata.normalize("NFKC", t)
    return WS.sub(" ", t).strip()                     # no lower-casing (left to models)

def build_input(row) -> str:                          # step 4: fixed order with field markers
    return " | ".join(f"{f.replace('_', ' ')}: {row[f]}" for f in TEXT_FIELDS)

def main():
    df = pd.read_csv(RAW)
    counts = {"start": len(df)}
    df = df[["job_id"] + TEXT_FIELDS + [LABEL]].copy()          # step 1
    for f in TEXT_FIELDS:
        df[f] = df[f].map(normalise)                            # step 2

    # Missingness per field by class (report table)
    miss = df.assign(**{f: df[f].eq("") for f in TEXT_FIELDS}).groupby(LABEL)[TEXT_FIELDS].mean()
    miss.to_csv(RESULTS / "p2_missingness_by_class.csv")

    df["text"] = df.apply(build_input, axis=1)
    df["n_chars"] = df[TEXT_FIELDS].apply(lambda r: len("".join(r)), axis=1)

    log = []                                                    # step 6
    bad_label = ~df[LABEL].isin([0, 1])
    short = df["n_chars"] < MIN_CHARS                           # step 5
    for reason, mask in [("invalid_label", bad_label), ("too_short", short & ~bad_label)]:
        log += [{"job_id": j, "reason": reason} for j in df.loc[mask, "job_id"]]
    df = df[~(bad_label | short)]
    counts["after_eligibility"] = len(df)

    pd.DataFrame(log, columns=["job_id", "reason"]).to_csv(RESULTS / "p2_exclusion_log.csv", index=False)
    df[["job_id", "text", LABEL]].to_csv(PROC / "clean.csv", index=False)
    pd.Series(counts).to_csv(RESULTS / "p2_record_counts.csv")
    print(counts)

if __name__ == "__main__":
    main()
