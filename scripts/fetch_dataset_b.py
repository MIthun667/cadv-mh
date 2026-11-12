#!/usr/bin/env python
from __future__ import annotations

from datasets import load_dataset

from _common import ROOT


REPOSITORY = "jsfactory/mental_health_reddit_posts"
REVISION = "e4ee1955e5173563708041f8a3a62b81fe703e38"
DESTINATION = ROOT.parent / "Data" / "data_1" / "mental_health_reddit_posts.csv"


def main() -> None:
    dataset = load_dataset(REPOSITORY, revision=REVISION, split="train")
    frame = dataset.to_pandas()[["body", "subreddit"]]
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DESTINATION, index=False)
    print(f"Dataset B fetched: {len(frame):,} rows written to {DESTINATION}")


if __name__ == "__main__":
    main()
