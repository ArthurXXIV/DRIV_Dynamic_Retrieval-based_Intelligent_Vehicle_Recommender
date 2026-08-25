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
DEFAULT_OUT_FULL = ROOT / "Data cleaning" / "Cleaned_data_full.csv"
REPORT = ROOT / "Data cleaning" / "cleaning_report.txt"

# Plausible ex-showroom range for an Indian car, in rupees. Anything outside is
# almost certainly a parse error -- an EMI figure (~Rs 25,000) or a typo.
PRICE_MIN, PRICE_MAX = 50_000, 200_000_000

# Spec section headers carry their own item count: "Engine Performance (8)".
# They are excluded from feature matching because they enumerate examples of
# the category rather than what a given car has.
SECTION_RE = re.compile(r"^(.*?)\s*\((\d+)\)$")

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
     "Discontinued", "Fuel Type", "Mileage", "Transmission", "Engine",
     "Displacement (cc)", "Cylinders", "Cylinder Layout",
     "Valves per Cylinder", "Valve Train",
     "Cluster Type", "Cluster Size", "Heads Up Display", "Tachometer",
     "Display"]
    + [f for f in BOOL_FIELDS]
    + ["Instrument Cluster", "Combined Description"]
)


def unslug(text):
    """'carens-clavis' -> 'Carens Clavis'; 'db11' -> 'DB11'.

    Alphanumeric tokens are model codes (DB11, EV6, 718, X1), so they are
    upper-cased rather than title-cased into 'Db11'.
    """
    if not isinstance(text, str):
        return text
    words = re.sub(r"[-_]+", " ", text).strip().split()
    return " ".join(
        w.upper() if (any(c.isdigit() for c in w) and any(c.isalpha() for c in w))
        else w.title()
        for w in words)


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
        def searchable(text):
            """Drop section headers before matching.

            Headers enumerate examples of their category -- "Driver Assistance
            (Park Assist, Cruise Control, etc.) (3)" -- so searching them marks
            every car with that section as having cruise control. Only the
            item lines state what a car actually has.
            """
            keep = [ln for ln in str(text).splitlines()
                    if not SECTION_RE.match(ln.strip())]
            return "\n".join(keep)

        haystack = (df[raw_cols].fillna("").map(searchable)
                    .agg(" ".join, axis=1).str.lower())

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


NA = "na"

# CarWale's SEO meta description is boilerplate ("Get price, mileage and
# available offers in India for X at CarWale") and says nothing about the car,
# so it is noise in the embedding text. We detect it and synthesise a real
# description from the structured fields instead.
BOILERPLATE_RE = re.compile(
    r"get price, mileage and available offers|at carwale", re.I)


def parse_engine(text):
    """'6749 cc, 12 Cylinders In V Shape, 4 Valves/Cylinder, DOHC' -> columns.

    Fixed comma-separated format, so a regex per field is exact and
    reproducible. Fields absent from a short string ('2800 cc') stay 'na'.
    """
    out = {"Displacement (cc)": NA, "Cylinders": NA, "Cylinder Layout": NA,
           "Valves per Cylinder": NA, "Valve Train": NA}
    if not isinstance(text, str):
        return out

    m = re.search(r"(\d+)\s*cc\b", text, re.I)
    if m:
        out["Displacement (cc)"] = int(m.group(1))

    m = re.search(r"(\d+)\s*Cylinders?", text, re.I)
    if m:
        out["Cylinders"] = int(m.group(1))

    m = re.search(r"Cylinders?\s+(Inline|In\s+([A-Z])\s+Shape|Flat|Boxer|Rotary)",
                  text, re.I)
    if m:
        out["Cylinder Layout"] = (m.group(2).upper() if m.group(2)
                                  else m.group(1).title())

    m = re.search(r"(\d+)\s*Valves?\s*/\s*Cylinder", text, re.I)
    if m:
        out["Valves per Cylinder"] = int(m.group(1))

    m = re.search(r"\b(DOHC|SOHC|OHV|OHC)\b", text, re.I)
    if m:
        out["Valve Train"] = m.group(1).upper()
    return out


def parse_cluster(text):
    """Pull the useful facts out of the Instrument Cluster blob.

    The raw value is the whole cluster section joined with pipes, so the
    individual facts have to be picked out by name.
    """
    out = {"Cluster Type": NA, "Cluster Size": NA,
           "Heads Up Display": "no", "Tachometer": NA}
    if not isinstance(text, str):
        return out

    m = re.search(r"(Analogue\s*-\s*Digital|Digital|Analogue|TFT|LCD)\s+"
                  r"Instrument Cluster", text, re.I)
    if m:
        kind = m.group(1)
        kind = kind.upper() if kind.upper() in ("TFT", "LCD") else kind.title()
        out["Cluster Type"] = re.sub(r"\s*-\s*", " - ", kind)

    m = re.search(r"([\d.]+)\s*-?\s*inch", text, re.I)
    if m:
        out["Cluster Size"] = f"{m.group(1)}-inch"

    if re.search(r"Heads Up Display", text, re.I):
        out["Heads Up Display"] = ("Optional"
                                   if re.search(r"Heads Up Display[^|]*\(Optional\)",
                                                text, re.I) else "Yes")

    m = re.search(r"(Analogue|Digital)\s+Tachometer", text, re.I)
    if m:
        out["Tachometer"] = m.group(1).title()
    return out


def synthesise_description(row):
    """Build a factual description from the structured fields.

    Replaces CarWale's SEO boilerplate with something that actually describes
    the car, which is what gets embedded for semantic search.
    """
    name = " ".join(str(row.get(f, "")).strip()
                    for f in ("Brand", "Car", "Variant")).strip()
    bits = []

    fuel = row.get("Fuel Type")
    trans = row.get("Transmission")
    if _has(fuel) and _has(trans):
        bits.append(f"is a {fuel} car with {trans} transmission")
    elif _has(fuel):
        bits.append(f"is a {fuel} car")
    elif _has(trans):
        bits.append(f"has {trans} transmission")

    price, ptype = row.get("Price"), row.get("Price_Type")
    if pd.notna(price):
        label = f" ({ptype})" if _has(ptype) else ""
        bits.append(f"priced at Rs {int(price):,}{label}")

    cc, cyl = row.get("Displacement (cc)"), row.get("Cylinders")
    if _has(cc) and _has(cyl):
        bits.append(f"powered by a {cc} cc {cyl}-cylinder engine")
    elif _has(cc):
        bits.append(f"powered by a {cc} cc engine")

    mileage = row.get("Mileage")
    if pd.notna(mileage):
        bits.append(f"returning {mileage} kmpl")

    highlights = [f for f in ("Sunroof / Moonroof", "Ventilated Seats",
                              "Cruise Control", "Wireless Charger")
                  if str(row.get(f, "no")).lower() == "yes"]
    sentence = f"{name} " + ", ".join(bits) + "." if bits else f"{name}."
    if highlights:
        sentence += " Features include " + ", ".join(highlights).lower() + "."
    return sentence


def _has(v):
    return v is not None and str(v).strip().lower() not in ("", "na", "nan", "no", "-")


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
    ap.add_argument("--out", dest="dst", default=str(DEFAULT_OUT),
                    help="bot-ready subset: rows with a usable numeric price")
    ap.add_argument("--out-full", dest="dst_full", default=str(DEFAULT_OUT_FULL),
                    help="everything scraped, discontinued and unpriced included")
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

    out["Discontinued"] = df.get("Discontinued", "no")
    out["Description"] = df.get("Description")
    out["Price_Type"] = df.get("Price_Type", "unknown")
    out["Price"] = df.get("Price").map(to_rupees) if "Price" in df else np.nan

    out = pd.concat([out, resolve_value_fields(df, report)], axis=1)
    report.append("")
    report.append("BOOLEAN FEATURES")
    out = pd.concat([out, resolve_bool_fields(df, report)], axis=1)

    out["Mileage"] = out["Mileage"].map(to_mileage)

    # --- split composite fields into real columns ------------------------
    report.append("")
    report.append("FIELD SPLITTING")
    engine_cols = out["Engine"].map(parse_engine).apply(pd.Series)
    cluster_cols = out["Instrument Cluster"].map(parse_cluster).apply(pd.Series) \
        if "Instrument Cluster" in out.columns else pd.DataFrame(index=out.index)
    out = pd.concat([out, engine_cols, cluster_cols], axis=1)

    # Parse coverage is the validator: a field that silently stops parsing
    # shows up here as a collapsed percentage.
    for col in list(engine_cols.columns) + list(cluster_cols.columns):
        filled = (~out[col].astype(str).str.lower().isin([NA, "nan", "no"])).mean() * 100
        flag = "  <-- LOW" if filled < 20 else ""
        report.append(f"  {col}: {filled:.0f}% parsed{flag}")

    # Replace CarWale's SEO boilerplate with a factual description.
    desc = out["Description"].astype(str)
    boiler = desc.str.contains(BOILERPLATE_RE, na=False)
    if boiler.any():
        report.append(f"  Description: {100*boiler.mean():.0f}% was SEO "
                      "boilerplate, synthesised from structured fields instead")
        out["Meta_Description"] = out["Description"]
        out.loc[boiler, "Description"] = out[boiler].apply(
            synthesise_description, axis=1)

    # --- two outputs ----------------------------------------------------
    # full : everything scraped, including discontinued and unpriced cars.
    #        This is the published Kaggle dataset -- completeness is the point.
    # bot  : only rows the recommender can actually use. app.py needs a numeric
    #        price, so a row without one would just be dropped downstream.
    report.append("")
    report.append("VALIDATION")
    problems = []

    out["Combined Description"] = out.apply(combine_description, axis=1)
    out = out[[c for c in CANONICAL_ORDER if c in out.columns]]
    full = out.copy()

    bot = out.dropna(subset=["Price"])
    unpriced = len(out) - len(bot)
    if unpriced:
        report.append(f"  {unpriced} rows have no parseable price "
                      "(kept in full, excluded from the bot dataset)")

    bad = bot[(bot["Price"] < PRICE_MIN) | (bot["Price"] > PRICE_MAX)]
    if len(bad):
        pct = 100 * len(bad) / len(bot)
        report.append(f"  dropped {len(bad)} rows ({pct:.0f}%) with implausible "
                      f"prices, e.g. {bad['Price'].head(3).tolist()}")
        bot = bot[(bot["Price"] >= PRICE_MIN) & (bot["Price"] <= PRICE_MAX)]
        # Only a widespread failure means the parser is broken; a few odd rows
        # are just cars CarWale prices unusually.
        if pct > 10:
            problems.append(f"{pct:.0f}% of rows had implausible prices -- "
                            "the price parser is likely broken, not the data")

    types = bot["Price_Type"].dropna().unique().tolist()
    if len(types) > 1:
        report.append(f"  MIXED price types {types} -- on-road and ex-showroom "
                      "are not comparable; filter on Price_Type before ranking")
    if args.ex_showroom_only:
        kept = bot[bot["Price_Type"] == "ex-showroom"]
        report.append(f"  --ex-showroom-only: kept {len(kept)}/{len(bot)} rows")
        bot = kept

    if "Discontinued" in full.columns:
        n_disc = (full["Discontinued"].astype(str).str.lower() == "yes").sum()
        report.append(f"  discontinued models: {n_disc} rows (full only)")

    mileage_null = bot["Mileage"].isna().mean() * 100 if len(bot) else 100
    report.append(f"  Mileage missing (bot dataset): {mileage_null:.0f}%")
    if mileage_null > 60:
        problems.append(f"Mileage missing for {mileage_null:.0f}% of bot rows; "
                        "app.py drops rows without it")

    if bot.empty:
        problems.append("no rows survived cleaning into the bot dataset")

    dst, dst_full = Path(args.dst), Path(args.dst_full)
    if dst.exists() and not args.force and not bot.empty:
        prev = pd.read_csv(dst)
        if len(bot) < 0.5 * len(prev):
            problems.append(f"row count collapsed: {len(bot)} vs {len(prev)} "
                            "previously -- refusing to overwrite (use --force)")

    report.append("")
    report.append(f"full : {len(full)} rows x {len(full.columns)} cols -> {dst_full.name}")
    report.append(f"bot  : {len(bot)} rows x {len(bot.columns)} cols -> {dst.name}")
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

    for frame, path in ((full, dst_full), (bot, dst)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    print(f"\nOK -- full={len(full)} rows, bot={len(bot)} rows")


if __name__ == "__main__":
    main()
