# CarWale scraper

Collects model/trim data from carwale.com into `output.csv` for the RAG index.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Chrome must be installed; Selenium Manager fetches the matching driver.

## Run

```bash
# Verify extraction still works against the live site (do this first, monthly)
./.venv/bin/python "Web Scraping/data_extractor.py" --selfcheck

# Full crawl
./.venv/bin/python "Web Scraping/data_extractor.py"

# Small trial
./.venv/bin/python "Web Scraping/data_extractor.py" --brands kia-cars --limit-models 3

# Continue an interrupted crawl
./.venv/bin/python "Web Scraping/data_extractor.py" --resume
```

A full crawl takes a few hours -- there is a randomised 1-2s pause per page to
stay polite. Progress is checkpointed to `car_data.json` after every model, so
`--resume` picks up where it stopped.

## Why it broke before, and what changed

The original scraper selected on CSS classes like `o-cpnuEd o-SoIQT o-cJrNdO`.
Those are build-generated hashes that CarWale rotates on every deploy, so the
selectors matched nothing. Worse, every failure was caught by a bare `except`
and the script still wrote a CSV -- so it looked like it ran fine while
producing empty data.

This version:

- **never selects on a class name.** Discovery is anchored on URL shape;
  extraction on `data-index`, semantic tags, and visible label text.
- **does not enumerate spec fields.** Whatever label/value pairs the page
  publishes become columns, so new fields appear automatically and renamed ones
  surface under the new name instead of silently going null. Column count
  therefore varies month to month by design.
- **self-validates the parse.** Each section header carries its own item count
  (`Engine Performance (8)`). If pairing yields exactly that many pairs the
  layout is intact and real columns are emitted; otherwise the section's raw
  text is kept verbatim rather than emitting misaligned garbage.
- **keeps full-fidelity text** in the `*_Raw` columns regardless, so the RAG
  index always has complete content even if parsing drifts.
- **fails loudly.** `--selfcheck` runs automatically before every crawl and
  aborts if extraction is broken, and the script refuses to overwrite the
  previous CSV if it collected less than half as many rows (writes
  `output.suspect.csv` instead). Override with `--force`.

## When CarWale redesigns again

Run `--selfcheck`. It prints a PASS/FAIL per extraction stage, so the broken
tier is obvious. Only that tier needs attention -- the others keep working.

## Price_Type -- read this before using Price

CarWale uses several price layouts, and some trim pages lead with the **on-road**
price while others show **ex-showroom**. These are not comparable (on-road
includes registration and insurance, so it runs 15-25% higher). The scraper
therefore records which kind it found in the `Price_Type` column and never
guesses.

Filter on `Price_Type` before any budget-based ranking. If a run reports mixed
types it logs a warning. Rows where only an on-road price was available (some
trims collapse their ex-showroom table behind a "View N Variants" control) are
still useful, but must not be compared directly against ex-showroom rows.

The scraper also never falls back to "the first price on the page" -- that lands
on the EMI figure, which is how an earlier version recorded a monthly instalment
of `Rs. 24,835` as a car's price.

## Known limitations

- Prices are city-dependent; CarWale defaults to Mumbai.
- Superseded models (year-range or `old-generation` slugs) are skipped, since
  they carry no current price. Pass `--include-discontinued` to keep them.
- Discontinued models expose no trims and are skipped (logged at DEBUG).
- Structured-data tiers (`Meta_*`, `OG_*`, `LD_*`) are populated only where
  CarWale publishes them; `LD_Price` is often absent.
