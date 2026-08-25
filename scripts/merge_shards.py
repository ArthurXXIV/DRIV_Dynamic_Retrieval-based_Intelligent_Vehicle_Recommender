"""Merge per-shard scraper CSVs into one dataset.

CI splits the crawl across parallel shards (a full run is ~16h, past the 6h
GitHub Actions job limit). Each shard emits its own CSV. Because the scraper's
columns are deliberately dynamic, shards can legitimately have different column
sets -- pandas.concat unions them and fills the gaps with NaN, which is exactly
what we want.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="shard CSVs to merge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-rows", type=int, default=200,
                    help="fail if the merged result is smaller than this")
    args = ap.parse_args()

    frames = []
    for path in args.inputs:
        p = Path(path)
        if not p.exists():
            print(f"WARNING: missing shard {p}")
            continue
        df = pd.read_csv(p)
        print(f"  {p.name}: {len(df)} rows, {len(df.columns)} cols")
        frames.append(df)

    if not frames:
        sys.exit("No shard files found -- every scrape job must have failed.")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    if "URL" in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset=["URL"])
        if before - len(merged):
            print(f"  dropped {before - len(merged)} duplicate URLs")

    print(f"Merged: {len(merged)} rows x {len(merged.columns)} cols "
          f"from {len(frames)} shards")

    if len(merged) < args.min_rows:
        sys.exit(f"Merged dataset has only {len(merged)} rows "
                 f"(expected >= {args.min_rows}). Refusing to publish.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
