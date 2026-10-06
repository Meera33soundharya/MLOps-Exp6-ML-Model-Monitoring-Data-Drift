"""Compare reference and current CSV data and write a drift report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scipy.stats import chi2_contingency, ks_2samp


NUMERIC_FEATURES = ("age", "monthly_spend", "support_tickets")
CATEGORICAL_FEATURES = ("plan", "region")
EXPECTED_FIELDS = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if tuple(reader.fieldnames or ()) != EXPECTED_FIELDS:
            expected = ", ".join(EXPECTED_FIELDS)
            raise ValueError(f"{path} must have these columns in order: {expected}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            for feature in EXPECTED_FIELDS:
                value = row.get(feature)
                if value is None or not value.strip():
                    raise ValueError(
                        f"{path}:{line_number} has an empty value for {feature!r}"
                    )
            rows.append(row)

    if not rows:
        raise ValueError(f"{path} contains no data rows")
    return rows


def numeric_values(rows: list[dict[str, str]], feature: str, path: Path) -> list[float]:
    values: list[float] = []
    for line_number, row in enumerate(rows, start=2):
        try:
            value = float(row[feature])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path}:{line_number} has an invalid value for {feature!r}"
            ) from error
        if not math.isfinite(value):
            raise ValueError(
                f"{path}:{line_number} has a non-finite value for {feature!r}"
            )
        values.append(value)
    return values


def compare_feature(
    reference: list[dict[str, str]],
    current: list[dict[str, str]],
    feature: str,
    kind: str,
    threshold: float,
    reference_path: Path,
    current_path: Path,
) -> dict[str, Any]:
    if kind == "numeric":
        reference_values = numeric_values(reference, feature, reference_path)
        current_values = numeric_values(current, feature, current_path)
        test_result = ks_2samp(reference_values, current_values)
        test_name = "kolmogorov_smirnov"
        statistic = float(test_result.statistic)
        p_value = float(test_result.pvalue)
    else:
        reference_counts: dict[str, int] = {}
        current_counts: dict[str, int] = {}
        for row in reference:
            reference_counts[row[feature]] = reference_counts.get(row[feature], 0) + 1
        for row in current:
            current_counts[row[feature]] = current_counts.get(row[feature], 0) + 1

        categories = sorted(reference_counts.keys() | current_counts.keys())
        if len(categories) < 2:
            statistic, p_value = 0.0, 1.0
        else:
            observed = [
                [reference_counts.get(category, 0) for category in categories],
                [current_counts.get(category, 0) for category in categories],
            ]
            statistic, p_value, _, _ = chi2_contingency(observed)
        test_name = "chi_square"

    return {
        "feature": feature,
        "kind": kind,
        "test": test_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < threshold),
    }


def render_report(summary: dict[str, Any], output_path: Path) -> None:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(result['feature'])}</td>"
        f"<td>{html.escape(result['test'])}</td>"
        f"<td>{result['statistic']:.6g}</td>"
        f"<td>{result['p_value']:.6g}</td>"
        f"<td>{'Yes' if result['drift_detected'] else 'No'}</td>"
        "</tr>"
        for result in summary["features"]
    )
    status = "Drift detected" if summary["drift_detected"] else "No drift detected"
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Data Drift Report</title>
  <style>
    body {{ color: #1f2937; font: 16px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 960px; padding: 0 1rem; }}
    h1 {{ margin-bottom: .25rem; }}
    .status {{ background: {'#fee2e2' if summary['drift_detected'] else '#dcfce7'}; border-radius: .4rem; display: inline-block; padding: .45rem .75rem; }}
    table {{ border-collapse: collapse; margin-top: 1.5rem; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d1d5db; padding: .65rem; text-align: left; }}
    th {{ background: #f3f4f6; }}
    .meta {{ color: #4b5563; }}
  </style>
</head>
<body>
  <h1>Data Drift Report</h1>
  <p class="status">{status}</p>
  <p class="meta">Generated {html.escape(summary['generated_at'])} ·
     {summary['reference_rows']} reference rows · {summary['current_rows']} current rows ·
     significance threshold {summary['threshold']}</p>
  <table>
    <thead><tr><th>Feature</th><th>Test</th><th>Statistic</th><th>p-value</th><th>Drift</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def detect_drift(
    reference_path: Path,
    current_path: Path,
    summary_path: Path,
    report_path: Path,
    threshold: float = 0.05,
) -> dict[str, Any]:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1 (exclusive)")

    reference = read_csv(reference_path)
    current = read_csv(current_path)
    features = [
        compare_feature(
            reference,
            current,
            feature,
            kind,
            threshold,
            reference_path,
            current_path,
        )
        for kind, feature_names in (
            ("numeric", NUMERIC_FEATURES),
            ("categorical", CATEGORICAL_FEATURES),
        )
        for feature in feature_names
    ]
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "drifted_features": sum(result["drift_detected"] for result in features),
        "drift_detected": any(result["drift_detected"] for result in features),
        "features": features,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    render_report(summary, report_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("data/reference.csv"))
    parser.add_argument("--current", type=Path, default=Path("data/current.csv"))
    parser.add_argument("--summary", type=Path, default=Path("drift_summary.json"))
    parser.add_argument("--report", type=Path, default=Path("drift_report.html"))
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    summary = detect_drift(
        args.reference,
        args.current,
        args.summary,
        args.report,
        args.threshold,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
