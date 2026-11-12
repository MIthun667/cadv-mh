#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dataset_a", "dataset_b"], required=True)
    args = parser.parse_args()
    records = []
    for path in (ROOT / "experiments" / args.dataset).glob("**/metrics.json"):
        record = pd.read_json(path, typ="series").to_dict()
        record["source"] = str(path.relative_to(ROOT))
        records.append(record)
    destination = ROOT / "experiments" / args.dataset / f"{args.dataset}_results.csv"
    pd.DataFrame(records).to_csv(destination, index=False)
    print(f"Results table saved to: {destination}")


if __name__ == "__main__":
    main()
