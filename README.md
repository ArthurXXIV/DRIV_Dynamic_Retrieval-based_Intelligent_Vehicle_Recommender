# DRIV

**D**ynamic **R**etrieval-based **I**ntelligent **V**ehicle recommender.

A RAG car recommender for the Indian market, built on a self-maintaining
data pipeline. It scrapes CarWale, publishes a cleaned dataset to Kaggle
every month via GitHub Actions, and serves recommendations through a
Streamlit chatbot backed by FAISS retrieval and Gemini generation.

**Dataset:** [Indian Cars Dataset on Kaggle](https://www.kaggle.com/datasets/atharvanilawar/indian-cars-dataset)

## Architecture

```
scrape → merge → clean → publish (Kaggle)
                      └→ embed → app.py (Streamlit + FAISS + Gemini)
```

Each stage is a separate artifact with its own contract.

| Component | Role |
|---|---|
| `Web Scraping/data_extractor.py` | Discovers brands, models and trims; extracts specs |
| `Data cleaning/Clean_Data.py` | Owns the output schema; emits the full and bot datasets |
| `scripts/merge_shards.py` | Unions the parallel CI shards |
| `scripts/build_embeddings.py` | RoBERTa encoding plus FAISS index construction |
| `app.py` | Streamlit chatbot |
| `.github/workflows/update-dataset.yml` | Monthly automated refresh |

## The dataset

2,614 rows across 34 columns, covering 40 brands, 735 models and 1,261
variants. 1,837 currently sold, 777 discontinued.

Coverage is uneven by design, because the source is real listing data:

| Field group | Fill |
|---|---|
| Identifiers, features, engine internals | 100% |
| Fuel type, transmission | 96.6% |
| Engine | 79.7% |
| Instrument cluster | 76.2% |
| Mileage | 58.2% |
| Price | 57.7% |

Discontinued models account for much of the price gap, since delisted
cars no longer carry a current figure.

**Two caveats for consumers of the data.** `Price` mixes two bases:
1,054 rows are ex-showroom and 454 are on-road, and on-road includes RTO,
insurance and road tax. Filter on `Price_Type` before any budget
comparison or ranking. Separately, `Display` fills at only 8.8%, which
is low enough that it is more likely a field-matching gap than genuine
absence.

## Design notes

**Extraction is layered by durability.** Structured data first (JSON-LD,
OpenGraph, meta tags), then generic key/value harvesting, then visible-
label anchoring. Nothing selects on a CSS class, because CarWale rotates
build-generated class hashes on every deploy. Spec fields are not
enumerated, so renames and additions flow through without a code change.
The first run after the rewrite auto-discovered a HIGHLIGHTS block the
previous hardcoded version would have dropped.

**The schema lives in the cleaner, not the scraper.** The scraper
captures whatever exists, and its columns vary between runs by design.
The cleaner maps that onto a fixed contract through an alias map, so a
site-side rename is one string added to a list.

**Failures are loud.** A `--selfcheck` pass runs before every crawl and
aborts on failure. The cleaning report gives per-field parse coverage.
Row-count and price-plausibility guards block publication of bad data.

**CI shards the crawl.** 44 brands are split across 8 parallel shards by
stride, then merged, cleaned and published. A serial crawl runs long
enough to exceed GitHub's six hour job cap; sharded, the full
scrape-to-publish cycle completes in about 24 minutes.

## Why the rebuild

The original scraper had been returning nothing useful for roughly two
years without anyone noticing. Every class-based selector had gone dead
after site redesigns, every exception was caught by a bare `except`, and
the script wrote a CSV regardless of outcome. The scheduled workflow
failed silently too, pointing at a path that no longer existed and never
installing Chrome.

Bugs found during the rebuild, all surfaced by running the pipeline
rather than by reading it:

- Price extraction returned the monthly EMI figure
- Cars priced in crore parsed as single-digit rupee values, because the
  unit was optional in the regex
- Stale element references would have crashed a full crawl partway
- Failed runs re-uploaded the previously committed CSV, so a broken run
  looked successful
- Section headers matched as features, giving a 12 percentage point
  false positive rate on feature flags
- The description field was almost entirely SEO boilerplate and near
  identical across cars, making it weak as an embedding source
- Mileage was never converted to numeric, which would have caused the
  app to drop every row

The scraper is not redesign-proof and nothing can be. What changed is
that the next break will be loud, localized and named in a report.

## Setup

```bash
git clone https://github.com/ArthurXXIV/DRIV.git
cd DRIV
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key"
streamlit run app.py
```

Embedding generation has its own dependency set in
`requirements-embeddings.txt`. See `PIPELINE.md` for the full
scrape-to-publish flow.

`kaggle` is pinned to 2.2.4 for reproducibility. That version dropped
username/key auth, which is why the pipeline authenticates with a token
(`KAGGLE_API_TOKEN`). The pin exists so an unpinned install cannot
silently jump versions again.

## Stack

Python, Selenium, Sentence Transformers (all-roberta-large-v1), FAISS,
Google Gemini, Streamlit, pandas, GitHub Actions.

## License

MIT.
