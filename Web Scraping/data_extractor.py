"""
CarWale scraper -- layout-resilient rewrite.

The original version selected on build-generated class hashes
(`o-cpnuEd o-SoIQT ...`). CarWale rotates those on every deploy, so the scraper
died and -- because every failure was swallowed by a bare `except` -- still
wrote a CSV, which is why the breakage went unnoticed.

This version never selects on a class name. It extracts in tiers, most durable
first, so a redesign degrades instead of dying:

  1. Structured data (JSON-LD, OpenGraph, meta) -- maintained for SEO, so it
     survives visual redesigns.
  2. Generic key/value harvesting -- we do NOT enumerate the ~140 spec fields.
     Whatever label/value pairs the page publishes become columns. New fields
     appear automatically; renamed ones show up under their new name instead of
     silently going null.
  3. Label-anchored lookup on visible human text ("Ex-Showroom Price"), which
     changes far less often than markup.

Self-validating parse: each spec section header carries its own item count
("Engine Performance (8)"). If the parse yields exactly that many pairs the
layout is intact and we emit columns; otherwise we keep the section's raw text
instead of emitting garbage. Raw section text is retained either way, so the
downstream RAG index always has full-fidelity content.

Usage:
    python data_extractor.py --selfcheck          # verify against live site
    python data_extractor.py                      # full crawl
    python data_extractor.py --brands tata-cars --limit-models 2
    python data_extractor.py --resume             # continue an interrupted run
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger as logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

BASE = "https://www.carwale.com"
HERE = Path(__file__).resolve().parent

# The unit is REQUIRED. It used to be optional, which let a match succeed on
# just "Rs. 2.04" when the unit sat outside the matched region -- a Rs 2.04
# crore Porsche was then read as two rupees. Requiring it means such a match
# fails and get_price falls through to the next strategy instead.
PRICE_UNIT = r"(?:Lakhs?|Crores?|Cr\b|L\b)"
PRICE_RE = r"Rs\.\s*[\d.,]+(?:\s*-\s*[\d.,]+)?\s*" + PRICE_UNIT
SECTION_RE = re.compile(r"^(.*?)\s*\((\d+)\)$")

# Discontinued models keep their pages up but carry no current price, so they
# add hours of crawling and rows the recommender can never actually suggest.
# CarWale marks them in the slug with a production year-range or an explicit
# "old-generation" suffix.
DISCONTINUED_RE = re.compile(
    r"-(?:19|20)\d{2}-(?:19|20)\d{2}$"   # carens-2023-2024
    r"|old-generation"                    # old-generation-seltos-2024
    r"|^(?:19|20)\d{2}-")                 # 1991-sierra-1991-2003

# Noise lines that appear inside spec blocks but are UI chrome, not data.
NOISE = ("Report incorrect", "View all", "Show more", "Show less", "Collapse",
         "Compare", "Expand", "Read More")

# Sub-pages that share a model's URL prefix but are not models/trims.
NOT_A_MODEL = {
    "images", "news", "expert-reviews", "videos", "reviews", "mileage",
    "colours", "360-view", "used", "dealers", "offers", "specifications",
    "variants", "compare", "on-road-price", "service-cost", "faqs",
    "price-in-india", "user-reviews", "road-test", "features", "gallery",
    "gst-price", "gallery-videos", "emi-calculator", "insurance",
}

# Fuel/transmission filter pages sit at trim depth but are not trims -- they
# re-list the same variants and would duplicate every row.
FILTER_SLUGS = {
    "automatic", "manual", "petrol", "diesel", "cng", "electric", "hybrid",
    "ev", "lpg", "amt", "dct", "cvt", "4x4", "awd", "4x2",
}

BRANDS = [
    'maruti-suzuki-cars', 'tata-cars', 'kia-cars', 'toyota-cars', 'bmw-cars',
    'hyundai-cars', 'mahindra-cars', 'honda-cars', 'mg-cars', 'skoda-cars',
    'jaguar-cars', 'audi-cars', 'jeep-cars', 'renault-cars', 'porsche-cars',
    'nissan-cars', 'rolls-royce-cars', 'byd-cars', 'citroen-cars',
    'lamborghini-cars', 'volvo-cars', 'ferrari-cars', 'ford-cars',
    'lexus-cars', 'bugatti-cars', 'tesla-cars', 'volkswagen-cars',
    'bentley-cars', 'isuzu-cars', 'lotus-cars', 'maserati-cars', 'mini-cars',
    'aston-martin-cars', 'mclaren-cars', 'mitsubishi-cars', 'land-rover-cars',
    'haval-cars', 'ora-cars', 'peugeot-cars', 'mercedes-benz-cars',
    'fisker-cars', 'force-motors-cars', 'pmv-cars', 'pravaig-cars',
]


def initialize_logger(logfile):
    logging.remove()
    fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    logging.add(sys.stderr, format=fmt, level="INFO")
    logging.add(logfile, format=fmt, level="DEBUG", rotation="10 MB",
                retention=5, enqueue=True, catch=True)


def make_driver(headless=True):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--log-level=3")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(60)
    return driver


def polite_get(driver, url, delay=(1.0, 2.2)):
    driver.get(url)
    try:
        WebDriverWait(driver, 20).until(
            ec.presence_of_element_located((By.TAG_NAME, "h1")))
    except Exception:
        pass
    time.sleep(random.uniform(*delay))


def is_404(driver):
    return "Page Not Found" in (driver.title or "")


# --------------------------------------------------------------------------
# Discovery -- anchored on URL shape, which is the most stable thing on the
# site (it is public, linked, and search-indexed).
# --------------------------------------------------------------------------

def _slugs_under(driver, prefix, drop_filters=False):
    found = {}
    hrefs = driver.execute_script(
        "return Array.from(document.querySelectorAll('a[href]'), a => a.href)")
    for href in hrefs or []:
        if not href or not href.startswith(BASE):
            continue
        path = href[len(BASE):].split("?")[0].split("#")[0]
        if not path.startswith(prefix):
            continue
        parts = [p for p in path[len(prefix):].split("/") if p]
        if len(parts) != 1:
            continue
        slug = parts[0]
        if slug in NOT_A_MODEL or slug.startswith("price-in-"):
            continue
        if drop_filters and slug in FILTER_SLUGS:
            continue
        found.setdefault(slug, BASE + path)
    return found


def get_models(driver, brand, include_discontinued=False):
    polite_get(driver, f"{BASE}/{brand}/")
    if is_404(driver):
        logging.error(f"{brand}: brand page 404")
        return {}
    models = _slugs_under(driver, f"/{brand}/")
    if not include_discontinued:
        dropped = [s for s in models if DISCONTINUED_RE.search(s)]
        for slug in dropped:
            models.pop(slug)
        if dropped:
            logging.debug(f"{brand}: skipped {len(dropped)} discontinued models")
    return models


def get_trims(driver, brand, model_slug):
    polite_get(driver, f"{BASE}/{brand}/{model_slug}/")
    if is_404(driver):
        return {}
    return _slugs_under(driver, f"/{brand}/{model_slug}/", drop_filters=True)


# --------------------------------------------------------------------------
# Tier 1: structured data
# --------------------------------------------------------------------------

def structured_data(driver):
    """Pull JSON-LD / OpenGraph / meta. Maintained for SEO, so it is the most
    redesign-resistant surface on the page."""
    out = {}
    try:
        out["Meta_Description"] = driver.find_element(
            By.CSS_SELECTOR, 'meta[name="description"]').get_attribute("content")
    except Exception:
        pass
    for prop, key in (("og:title", "OG_Title"), ("og:description", "OG_Description")):
        try:
            out[key] = driver.find_element(
                By.CSS_SELECTOR, f'meta[property="{prop}"]').get_attribute("content")
        except Exception:
            pass

    faqs = []
    for script in driver.find_elements(
            By.CSS_SELECTOR, 'script[type="application/ld+json"]'):
        raw = script.get_attribute("innerHTML") or ""
        try:
            blob = json.loads(raw)
        except Exception:
            continue
        for node in (blob if isinstance(blob, list) else [blob]):
            if not isinstance(node, dict):
                continue
            offers = node.get("offers")
            if isinstance(offers, dict) and offers.get("price"):
                out.setdefault("LD_Price", str(offers["price"]))
            rating = node.get("aggregateRating")
            if isinstance(rating, dict) and rating.get("ratingValue"):
                out.setdefault("LD_Rating", str(rating["ratingValue"]))
            for q in node.get("mainEntity", []) or []:
                if isinstance(q, dict) and q.get("@type") == "Question":
                    ans = (q.get("acceptedAnswer") or {}).get("text", "")
                    faqs.append(f"Q: {q.get('name','')} A: {ans}")
    if faqs:
        out["FAQs"] = "\n".join(faqs)
    return out


# --------------------------------------------------------------------------
# Tier 3: label-anchored price
# --------------------------------------------------------------------------

def get_price(driver, title=None):
    """Return (price, price_type).

    CarWale uses at least three price layouts across models, and critically the
    headline figure is not always the same *kind* of price -- some trim pages
    lead with the on-road price instead of ex-showroom. Silently mixing the two
    would corrupt any budget-based recommendation, so the kind is always
    reported alongside the value and never guessed.

    We also never fall back to "the first Rs. on the page": that lands on the
    EMI figure and records a monthly instalment as the car's price.
    """
    # 1. Spec-table row (Tata-style).
    try:
        cell = driver.find_element(
            By.XPATH, "//th[contains(normalize-space(.),'Ex-Showroom Price')]"
                      "/following-sibling::td[1]")
        m = re.search(PRICE_RE, cell.get_attribute("innerText") or "")
        if m:
            return m.group(0).strip(), "ex-showroom"
    except Exception:
        pass

    body = driver.find_element(By.TAG_NAME, "body").get_attribute("innerText") or ""

    # 2. Inline "Ex-Showroom Price - Rs. X Lakh" (Kia Carens-style). Require a
    #    separator so we do not match the "VariantsEx-Showroom Price" table head.
    m = re.search(r"Ex-Showroom Price\s*[-\u2013:]\s*(" + PRICE_RE + ")", body)
    if m:
        return m.group(1).strip(), "ex-showroom"

    # 3. Variants table row for this trim (Carens Clavis-style), e.g.
    #    "Carens Clavis HTX\nCompare\nGet Offers\nRs. 13.40 - 15.98 Lakh".
    if title:
        name = re.escape(title.split(" ", 1)[-1])  # drop the brand word
        m = re.search(name + r"\n(?:[A-Za-z ]+\n){0,3}?(" + PRICE_RE + ")", body)
        if m:
            return m.group(1).strip(), "ex-showroom"

    # 4. Headline figure explicitly labelled as on-road. Recorded honestly as
    #    on-road rather than passed off as ex-showroom.
    m = re.search(r"(" + PRICE_RE + r")\s*\n[^\n]*On[- ]Road Price", body)
    if m:
        return m.group(1).strip(), "on-road"

    return None, None


# --------------------------------------------------------------------------
# Tier 2: generic key/value harvesting
# --------------------------------------------------------------------------

def parse_sections(text):
    """Split a spec block into {section_name: [lines]} using the "(N)" headers.

    Returns a list of (name, declared_count, lines) so callers can use the
    declared count as a checksum on their own parse.
    """
    sections, cur, count, buf = [], None, None, []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(NOISE):
            continue
        m = SECTION_RE.match(line)
        if m:
            if cur is not None:
                sections.append((cur, count, buf))
            cur, count, buf = m.group(1).strip(), int(m.group(2)), []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        sections.append((cur, count, buf))
    return sections


def harvest(text):
    """Turn a spec block into columns without knowing any field names.

    Alternating label/value sections become real columns. The section's own
    declared count is the checksum: if pairing yields exactly that many pairs
    the layout is intact; otherwise we keep the section verbatim in a single
    column rather than emitting misaligned garbage.
    """
    out = {}
    if not text:
        return out
    for name, declared, lines in parse_sections(text):
        pairs = {lines[i]: lines[i + 1] for i in range(0, len(lines) - 1, 2)}
        if declared and len(pairs) == declared:
            out.update(pairs)          # layout intact -> real columns
        elif lines:
            out[name] = " | ".join(lines)   # layout drifted -> keep raw
    return out


def spec_blocks(driver):
    """Every data-index block, keyed by its heading, whatever those turn out to
    be. We do not hardcode SPECIFICATION/SAFETY/FEATURES -- if CarWale adds a
    fourth block it is picked up automatically."""
    out = {}
    for el in driver.find_elements(By.CSS_SELECTOR, "div[data-index]"):
        text = (el.get_attribute("innerText") or "").strip()
        if text:
            out[text.splitlines()[0].strip()] = text
    return out


def scrape_trim(driver, url):
    polite_get(driver, url)
    if is_404(driver):
        return None
    try:
        title = driver.find_element(By.TAG_NAME, "h1").text.strip()
    except Exception:
        title = None
    price, price_type = get_price(driver, title)
    return {
        "url": url,
        "title": title,
        "price": price,
        "price_type": price_type,
        "structured": structured_data(driver),
        "blocks": spec_blocks(driver),
    }


def to_row(rec):
    # Column names deliberately match the original scraper's contract
    # (Car/Variant/Description) so Data cleaning/Clean_Data.py keeps working.
    structured = rec.get("structured") or {}
    row = {
        "Brand": rec["brand"].replace("-cars", ""),
        "Car": rec["model"],
        "Variant": rec["trim"],
        "Title": rec["title"],
        "Description": structured.get("Meta_Description") or structured.get("OG_Description"),
        "Price": rec["price"],
        "Price_Type": rec.get("price_type"),
        "Discontinued": "Yes" if rec.get("discontinued") else "no",
        "URL": rec["url"],
    }
    row.update(structured)
    for heading, text in (rec.get("blocks") or {}).items():
        row.update(harvest(text))
        # Full-fidelity text always survives, whatever the parse did.
        row[f"{heading.title()}_Raw"] = text
    return row


# --------------------------------------------------------------------------
# Health check
# --------------------------------------------------------------------------

def selfcheck(driver):
    """Prove the selectors still work before committing to a long crawl."""
    ok = True

    def report(label, got, detail=""):
        nonlocal ok
        ok &= bool(got)
        print(f"[{'PASS' if got else 'FAIL'}] {label}: {detail or got}")

    models = get_models(driver, "tata-cars")
    report("model discovery", models, f"{len(models)} models")
    trims = get_trims(driver, "tata-cars", "nexon") if models else {}
    report("trim discovery", trims, f"{len(trims)} trims")
    if not trims:
        return ok

    rec = scrape_trim(driver, sorted(trims.values())[0])
    report("title", rec and rec["title"], rec and rec["title"])
    report("price", rec and rec["price"],
           f"{rec and rec['price']} ({rec and rec['price_type']})")
    report("spec blocks", rec and rec["blocks"],
           ", ".join((rec.get("blocks") or {}).keys()))
    report("structured data", rec and rec["structured"],
           ", ".join((rec.get("structured") or {}).keys()) or "none")

    harvested = {}
    for text in (rec.get("blocks") or {}).values():
        harvested.update(harvest(text))
    report("harvested fields", len(harvested) > 20, f"{len(harvested)} fields")
    if harvested:
        sample = list(harvested.items())[:3]
        print("       e.g. " + "; ".join(f"{k}={v}"[:50] for k, v in sample))
    return ok


def main():
    ap = argparse.ArgumentParser(description="Scrape CarWale model/trim data.")
    ap.add_argument("--selfcheck", action="store_true",
                    help="verify extraction against the live site, then exit")
    ap.add_argument("--brands", nargs="*", default=BRANDS)
    ap.add_argument("--limit-models", type=int, default=None)
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="crawl only shard I of N (0-indexed). A full crawl runs "
                         "~16h, well past the 6h GitHub Actions job limit, so CI "
                         "splits the brand list across parallel shards.")
    ap.add_argument("--out", default=str(HERE / "output.csv"))
    ap.add_argument("--raw", default=str(HERE / "car_data.json"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-headless", action="store_true")
    ap.add_argument("--include-discontinued", action="store_true",
                    help="also crawl superseded models (year-range slugs)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the previous CSV even if this run looks worse")
    args = ap.parse_args()

    brands = args.brands
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        if not 0 <= i < n:
            sys.exit(f"--shard index {i} out of range for {n} shards")
        brands = brands[i::n]   # stride, so each shard gets a mix of big/small brands

    initialize_logger(str(HERE / "scraper.log"))
    driver = make_driver(headless=not args.no_headless)

    try:
        if args.selfcheck:
            good = selfcheck(driver)
            print("\nSELFCHECK:", "OK" if good else "EXTRACTION BROKEN")
            sys.exit(0 if good else 1)

        if not selfcheck(driver):
            logging.error("Self-check failed; aborting before the crawl. "
                          "The site layout has changed again.")
            sys.exit(1)

        rows, seen = [], set()
        raw_path = Path(args.raw)
        if args.resume and raw_path.exists():
            rows = json.loads(raw_path.read_text())
            seen = {r["url"] for r in rows}
            logging.info(f"Resuming with {len(seen)} rows already collected")

        logging.info(f"Crawling {len(brands)} brands"
                     + (f" (shard {args.shard})" if args.shard else ""))
        for brand in tqdm(brands, desc="Brands", position=0):
            try:
                models = get_models(driver, brand, args.include_discontinued)
            except Exception as e:
                logging.error(f"{brand}: model discovery failed: {e}")
                continue
            items = sorted(models.items())
            if args.limit_models:
                items = items[:args.limit_models]
            logging.info(f"{brand}: {len(items)} models")

            for model_slug, _ in tqdm(items, desc=brand, position=1, leave=False):
                try:
                    trims = get_trims(driver, brand, model_slug)
                except Exception as e:
                    logging.error(f"{brand}/{model_slug}: trim discovery failed: {e}")
                    continue
                if not trims:
                    logging.debug(f"{brand}/{model_slug}: no trims (likely discontinued)")
                    continue
                for trim_slug, url in sorted(trims.items()):
                    if url in seen:
                        continue
                    try:
                        rec = scrape_trim(driver, url)
                    except Exception as e:
                        logging.error(f"{url}: {e}")
                        continue
                    if not rec:
                        continue
                    rec.update(brand=brand, model=model_slug, trim=trim_slug,
                               discontinued=bool(DISCONTINUED_RE.search(model_slug)))
                    rows.append(rec)
                    seen.add(url)
                    logging.debug(f"OK {url} | {rec['price']}")
                raw_path.write_text(json.dumps(rows, indent=1))

        raw_path.write_text(json.dumps(rows, indent=1))
        df = pd.DataFrame([to_row(r) for r in rows])
        logging.info(f"Collected {len(df)} rows x {len(df.columns)} columns")
        if len(df):
            missing = int(df["Price"].isna().sum())
            pct = 100 * missing / len(df)
            level = logging.warning if pct > 20 else logging.info
            level(f"Price missing for {missing}/{len(df)} rows ({pct:.0f}%)")
            counts = df["Price_Type"].value_counts(dropna=True).to_dict()
            logging.info(f"Price types: {counts}")
            if len(counts) > 1:
                logging.warning(
                    "Mixed price types in output -- ex-showroom and on-road "
                    "figures are not comparable. Filter on Price_Type before "
                    "any budget-based ranking.")

        # Never let a broken run quietly replace good data -- the failure mode
        # that hid the original breakage.
        out_path = Path(args.out)
        if out_path.exists() and not args.force:
            prev = pd.read_csv(out_path)
            if len(df) < 0.5 * len(prev):
                logging.error(
                    f"Refusing to overwrite {out_path}: got {len(df)} rows vs "
                    f"{len(prev)} previously. Re-run with --force to override.")
                alt = out_path.with_suffix(".suspect.csv")
                df.to_csv(alt, index=False)
                logging.error(f"Wrote this run to {alt} for inspection instead.")
                sys.exit(1)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logging.info(f"Wrote {len(df)} rows x {len(df.columns)} cols -> {out_path}")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
