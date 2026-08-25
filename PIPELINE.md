# Data pipeline

    scrape  ->  merge  ->  clean  ->  Kaggle dataset
                                  ->  embeddings -> app.py

| Stage | Command | Output |
|---|---|---|
| Scrape | `python "Web Scraping/data_extractor.py"` | `Web Scraping/output.csv` |
| Clean | `python "Data cleaning/Clean_Data.py"` | `Data cleaning/Cleaned_data.csv` |
| Embed | `python scripts/build_embeddings.py` | `Cleaned_data_with_embeddings.csv` + FAISS index |
| App | `streamlit run app.py` | - |

Embeddings need extra deps: `pip install -r requirements-embeddings.txt`
(torch + sentence-transformers, ~2GB). They are **not** installed in CI, which
only scrapes, cleans and publishes.

## Automated monthly update

`.github/workflows/update-dataset.yml` runs at 03:00 UTC on the 1st of each
month, or on demand via **Actions -> Update Kaggle dataset -> Run workflow**.

A full crawl is ~2000 trim pages at roughly 30s each -- about 16 hours, far past
the 6-hour limit for a single GitHub-hosted job. So the workflow splits the 44
brands across 8 parallel shards (~2h each) using a stride, which spreads the big
brands one per shard. A `publish` job then merges the shards, cleans, and pushes
a new version of the Kaggle dataset.

`fail-fast: false` and `if: always()` mean one dead shard degrades the run
rather than killing it -- you get a slightly smaller dataset plus artifacts to
inspect, not nothing.

### Required repository secrets

Settings -> Secrets and variables -> Actions:

| Secret | Where to get it |
|---|---|
| `KAGGLE_USERNAME` | your Kaggle username |
| `KAGGLE_KEY` | Kaggle -> Settings -> API -> Create New Token (`kaggle.json`) |

The dataset published to is set in `kaggle_dataset/dataset-metadata.json`
(`atharvanilawar/indian-cars-dataset`). Each run adds a new *version* to that
dataset rather than replacing it, so previous months stay downloadable.

### Running the app

`app.py` reads `GOOGLE_API_KEY` from the environment -- never hardcode it:

    export GOOGLE_API_KEY=...
    streamlit run app.py

Data location can be overridden with `DRIV_DATA`.

## Testing safely

**Run workflow** takes three inputs that make testing cheap and non-destructive:

| Input | Use |
|---|---|
| `limit_models` = `1` | One model per brand -- finishes in minutes instead of ~16h |
| `publish` = `false` | Dry run: skips Kaggle entirely, just uploads artifacts |
| `dataset_id` | Publish to a different (e.g. private test) dataset |

Recommended first run: `limit_models=1`, `publish=false`. That exercises scrape
-> merge -> clean end to end and hands you the CSV as a downloadable artifact
without touching the live Kaggle dataset at all.

Then to test the publish path itself, create a private dataset on Kaggle and
pass its id as `dataset_id`. Only once that works should you let it write to
`atharvanilawar/indian-cars-dataset`.

Note: a *private GitHub repo* is the wrong place to test this. The Kaggle
dataset is the same target regardless of which repo publishes to it, so a
private repo isolates nothing -- and private repos are capped at 2,000 Actions
minutes/month while public repos are unlimited. One full crawl is ~960 minutes.
