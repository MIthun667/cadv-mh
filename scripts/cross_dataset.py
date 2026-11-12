#!/usr/bin/env python
from __future__ import annotations

import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["a_to_b", "b_to_a"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.parse_args()
    raise SystemExit(
        "Cross-dataset transfer is unavailable: Dataset A and the replacement "
        "OCD-vs-ADHD Dataset B do not share a valid label space."
    )


if __name__ == "__main__":
    main()
