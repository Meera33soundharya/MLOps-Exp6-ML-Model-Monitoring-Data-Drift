"""Generate reproducible reference and current datasets for drift monitoring."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


FIELDS = ("age", "monthly_spend", "support_tickets", "plan", "region")


def write_dataset(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_reference(rows: int, rng: random.Random) -> list[dict[str, object]]:
    plans = ("basic", "standard", "premium")
    plan_weights = (0.5, 0.35, 0.15)
    regions = ("north", "south", "east", "west")

    return [
        {
            "age": max(18, min(90, round(rng.gauss(42, 12)))),
            "monthly_spend": round(max(10, rng.gauss(65, 15)), 2),
            "support_tickets": min(8, rng.randint(0, 3) + (rng.random() < 0.1)),
            "plan": rng.choices(plans, weights=plan_weights, k=1)[0],
            "region": rng.choice(regions),
        }
        for _ in range(rows)
    ]


def make_current(rows: int, rng: random.Random) -> list[dict[str, object]]:
    plans = ("basic", "standard", "premium")
    plan_weights = (0.2, 0.4, 0.4)
    regions = ("north", "south", "east", "west")

    return [
        {
            "age": max(18, min(90, round(rng.gauss(45, 12)))),
            "monthly_spend": round(max(10, rng.gauss(78, 18)), 2),
            "support_tickets": min(10, rng.randint(0, 5) + (rng.random() < 0.25)),
            "plan": rng.choices(plans, weights=plan_weights, k=1)[0],
            "region": rng.choices(regions, weights=(0.15, 0.2, 0.45, 0.2), k=1)[0],
        }
        for _ in range(rows)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=500, help="rows to create in each CSV")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducible data")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    if args.rows < 2:
        parser.error("--rows must be at least 2")

    reference = make_reference(args.rows, random.Random(args.seed))
    current = make_current(args.rows, random.Random(args.seed + 1))
    write_dataset(args.output_dir / "reference.csv", reference)
    write_dataset(args.output_dir / "current.csv", current)
    print(f"Wrote {args.rows} rows to {args.output_dir / 'reference.csv'}")
    print(f"Wrote {args.rows} rows to {args.output_dir / 'current.csv'}")


if __name__ == "__main__":
    main()
