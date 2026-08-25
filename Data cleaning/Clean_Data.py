"""
Clean scraped CarWale data into a fixed schema for the embedding step.

Design note -- why this file looks the way it does:

The scraper deliberately does NOT emit a fixed set of columns. It captures
whatever carwale.com publishes, so its output changes shape whenever the site
does. That variability has to stop somewhere, and it stops here:

    scraper  -> fidelity: capture everything, whatever shape it is in
    cleaner  -> contract: emit a FIXED schema, whatever arrived

So this file owns the schema, not the scraper. Every downstream consumer
(embeddings, FAISS, app.py) can rely on CANONICAL_ORDER being stable forever.

Two mechanisms make that work:

1. Value fields resolve through an alias list. When CarWale renames "Fuel Type"
   to "Powertrain Type", you add one string to the alias list -- no code change.
2. Boolean features are matched by term presence against the `*_Raw` columns.
   CarWale moved features from key/value pairs ("Sunroof / Moonroof: Yes") to
   presence lists (a "Sunroof" bullet under "Sunroof & Windows"), so asking
   "does this term appear?" survives both formats.

Anything unresolved becomes a null and is named in the coverage report rather
than crashing the run or silently vanishing.

Usage:
    python "Data cleaning/Clean_Data.py"
    python "Data cleaning/Clean_Data.py" --ex-showroom-only
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = ROOT / "Web Scraping" / "output.csv"
DEFAULT_OUT = ROOT / "Data cleaning" / "Cleaned_data.csv"
REPORT = ROOT / "Data cleaning" / "cleaning_report.txt"

# Plausible ex-showroom range for an Indian car, in rupees. Anything outside is
# almost certainly a parse error -- an EMI figure (~Rs 25,000) or a typo.
PRICE_MIN, PRICE_MAX = 50_000, 200_000_000

# --- The contract -------------------------------------------------------
# Value fields: first alias found in the scraped columns wins.
VALUE_FIELDS = {
    "Fuel Type":    ["Fuel Type", "Fuel", "Powertrain Type", "Engine Type"],
    "Mileage":      ["Mileage", "Mileage (ARAI)", "ARAI Mileage",
                     "Fuel Efficiency", "Range"],
    "Transmission": ["Transmission", "Gearbox", "Transmission Type"],
    "Engine":       ["Engine", "Engine Type", "Displacement"],
    "Display":      ["Display", "Touchscreen Size", "Infotainment Display",
                     "Instrument Cluster"],
    "Instrument Cluster": ["Instrument Cluster", "Cluster Type"],
}

# Boolean features: matched by term presence against the *_Raw blocks.
BOOL_FIELDS = {
    "Sunroof / Moonroof": ["Sunroof", "Moonroof"],
    "Dashcam":            ["Dashcam", "Dash Cam"],
    "Rear AC":            ["Rear AC", "Rear Air Conditioner"],
    "Central Locking":    ["Central Locking"],
    "Cruise Control":     ["Cruise Control"],
    "Hill Hold Control":  ["Hill Hold"],
    "Ventilated Seats":   ["Ventilated Seat"],
    "Wireless Charger":   ["Wireless Charger", "Wireless Charging"],
    "Adjustable ORVMs":   ["Adjustable ORVM"],
    "Integrated (in-dash) Music System": ["Music System", "Infotainment",
                                          "Touchscreen"],
    "Speakers":           ["Speaker"],
}

IDENTITY = ["Brand", "Car", "Variant", "Description", "Price", "Price_Type"]

# The stable output schema. app.py and the embedding step depend on this.
CANONICAL_ORDER = (
    ["Brand", "Car", "Variant", "Description", "Price", "Price_Type",
     "Fuel Type", "Mileage", "Transmission", "Engine", "Display"]
    + [f for f in BOOL_FIELDS]
    + ["Instrument Cluster", "Combined Description"]
)


def unslug(text):
    """'carens-clavis' -> 'Carens Clavis'."""
    if not isinstance(text, str):
        return text
    return re.sub(r"[-_]+", " ", text).strip().title()


def resolve_value_fields(df, report):
    """Map scraped columns onto canonical names via aliases."""
    out = pd.DataFrame(index=df.index)
    for canonical, aliases in VALUE_FIELDS.items():
        for alias in aliases:
            if alias in df.columns and df[alias].notna().any():
                out[canonical] = df[alias]
                if alias != canonical:
                    report.append(f"  {canonical}: resolved via alias '{alias}'")
                break
        else:
            out[canonical] = np.nan
            report.append(f"  {canonical}: NOT FOUND -- emitted as null")
    return out


def resolve_bool_fields(df, report):
    """Detect features by term presence across every *_Raw block.

    Searching the combined raw text rather than a specific block means a feature
    still resolves if CarWale moves it between sections.
    """
    raw_cols = [c for c in df.columns if c.endswith("_Raw")]
    if not raw_cols:
        report.append("  WARNING: no *_Raw columns; boolean features unresolvable")
        haystack = pd.Series([""] * len(df), index=df.index)
    else:
        haystack = (df[raw_cols].fillna("").agg(" ".join, axis=1)
                    .str.lower())

    out = pd.DataFrame(index=df.index)
    for canonical, terms in BOOL_FIELDS.items():
        pattern = "|".join(re.escape(t.lower()) for t in terms)
        hits = haystack.str.contains(pattern, regex=True, na=False)
        out[canonical] = np.where(hits, "Yes", "no")
        rate = 100 * hits.mean()
        if rate == 0:
            report.append(f"  {canonical}: 0% match -- check the term list")
        else:
            report.append(f"  {canonical}: {rate:.0f}% of rows")
    return out


def to_rupees(text):
    """'Rs. 11.02 Lakh' -> 1102000. Ranges return the midpoint.

    A unit is mandatory. A bare 'Rs. 1.64' is never a literal 164 paise car --
    it means the Lakh/Crore token was lost during extraction, so returning NaN
    surfaces the row as missing rather than silently recording a crore car as
    two rupees.
    """
    if not isinstance(text, str):
        return np.nan
    s = text.replace("Estimated Price", "").replace("Rs.", "").replace(",", "").strip()
    low = s.lower()
    if re.search(r"crores?\b|\bcr\b", low):
        unit = 1e7
    elif re.search(r"lakhs?\b|\bl\b", low):
        unit = 1e5
    else:
        return np.nan
    nums = re.findall(r"\d+(?:\.\d+)?", re.sub(r"(?i)lakhs?|crores?|\bcr\b|\bl\b", "", s))
    if not nums:
        return np.nan
    vals = [float(n) * unit for n in nums[:2]]
    return round(sum(vals) / len(vals))


def to_mileage(text):
    """'21.79 kmpl' / 'User Reported: 15 kmpl' -> 21.79 / 15.0.

    app.py calls pd.to_numeric on this and drops NaN rows, so it must be a bare
    number -- the original pipeline left it as text.
    """
    if not isinstance(text, str):
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:kmpl|km/kg|km/l|km)", text, re.I)
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else np.nan


def combine_description(row):
    """Human-readable blob that gets embedded for semantic search."""
    parts = []
    if pd.notna(row.get("Description")):
        parts.append(str(row["Description"]))
    for col in CANONICAL_ORDER:
        if col in ("Description", "Combined Description", "Price_Type"):
            continue
        val = row.get(col)
        if pd.notna(val) and str(val).strip() not in ("", "no"):
            parts.append(f"{col}: {val}")
    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Clean scraped CarWale data.")
    ap.add_argument("--in", dest="src", default=str(DEFAULT_IN))
    ap.add_argument("--out", dest="dst", default=str(DEFAULT_OUT))
    ap.add_argument("--ex-showroom-only", action="store_true",
                    help="keep only ex-showroom prices (on-road figures are not "
                         "comparable for budget ranking)")
    ap.add_argument("--force", action="store_true",
                    help="write even if the run looks worse than the last one")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        sys.exit(f"Input not found: {src}\nRun the scraper first.")

    df = pd.read_csv(src)
    report = [f"Input : {src}", f"Rows  : {len(df)}",
              f"Cols  : {len(df.columns)}", "", "VALUE FIELDS"]

    missing_identity = [c for c in ("Brand", "Car", "Variant") if c not in df.columns]
    if missing_identity:
        sys.exit(f"Input is missing identity columns {missing_identity}; "
                 "the scraper's output contract has changed.")

    out = pd.DataFrame(index=df.index)
    out["Brand"] = df["Brand"].astype(str).str.replace("-", " ", regex=False).str.title()
    out["Car"] = df["Car"].map(unslug)
    out["Variant"] = df["Variant"].map(unslug)

    # Strip the brand out of the car name ("Kia Carens" -> "Carens").
    def strip_brand(row):
        name = str(row["Car"])
        for word in str(row["Brand"]).split():
            name = re.sub(rf"\b{re.escape(word)}\b", "", name, flags=re.I)
        return re.sub(r"\s+", " ", name).strip() or str(row["Car"])
    out["Car"] = out.apply(strip_brand, axis=1)

    out["Description"] = df.get("Description")
    out["Price_Type"] = df.get("Price_Type", "unknown")
    out["Price"] = df.get("Price").map(to_rupees) if "Price" in df else np.nan

    out = pd.concat([out, resolve_value_fields(df, report)], axis=1)
    report.append("")
    report.append("BOOLEAN FEATURES")
    out = pd.concat([out, resolve_bool_fields(df, report)], axis=1)

    out["Mileage"] = out["Mileage"].map(to_mileage)

    # --- deterministic validation gate ---------------------------------
    report.append("")
    report.append("VALIDATION")
    problems = []

    before = len(out)
    out = out.dropna(subset=["Price"])
    if before - len(out):
        report.append(f"  dropped {before - len(out)} rows with unparseable Price")

    bad = out[(out["Price"] < PRICE_MIN) | (out["Price"] > PRICE_MAX)]
    if len(bad):
        pct = 100 * len(bad) / len(out)
        report.append(f"  dropped {len(bad)} rows ({pct:.0f}%) with implausible "
                      f"prices, e.g. {bad['Price'].head(3).tolist()}")
        out = out[(out["Price"] >= PRICE_MIN) & (out["Price"] <= PRICE_MAX)]
        # Only a widespread failure means the parser is broken; a few odd rows
        # are just cars CarWale prices unusually.
        if pct > 10:
            problems.append(f"{pct:.0f}% of rows had implausible prices -- "
                            "the price parser is likely broken, not the data")

    types = out["Price_Type"].dropna().unique().tolist()
    if len(types) > 1:
        report.append(f"  MIXED price types {types} -- on-road and ex-showroom "
                      "are not comparable; filter on Price_Type before ranking")
    if args.ex_showroom_only:
        kept = out[out["Price_Type"] == "ex-showroom"]
        report.append(f"  --ex-showroom-only: kept {len(kept)}/{len(out)} rows")
        out = kept

    mileage_null = out["Mileage"].isna().mean() * 100
    report.append(f"  Mileage missing: {mileage_null:.0f}%")
    if mileage_null > 60:
        problems.append(f"Mileage missing for {mileage_null:.0f}% of rows; "
                        "app.py drops rows without it")

    if out.empty:
        problems.append("no rows survived cleaning")

    out["Combined Description"] = out.apply(combine_description, axis=1)
    out = out[[c for c in CANONICAL_ORDER if c in out.columns]]

    dst = Path(args.dst)
    if dst.exists() and not args.force and not out.empty:
        prev = pd.read_csv(dst)
        if len(out) < 0.5 * len(prev):
            problems.append(f"row count collapsed: {len(out)} vs {len(prev)} "
                            "previously -- refusing to overwrite (use --force)")

    report.append("")
    report.append(f"Output rows: {len(out)}  cols: {len(out.columns)}")
    if problems:
        report.append("")
        report.append("PROBLEMS")
        report.extend(f"  - {p}" for p in problems)

    text = "\n".join(report)
    REPORT.write_text(text)
    print(text)

    if problems:
        print(f"\nFAILED -- {len(problems)} problem(s). Report: {REPORT}")
        sys.exit(1)

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    print(f"\nOK -- wrote {len(out)} rows x {len(out.columns)} cols -> {dst}")


if __name__ == "__main__":
    main()
