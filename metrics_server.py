"""Run drift checks periodically and expose their results to Prometheus."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from prometheus_client import Gauge, start_http_server

from drift_detection import detect_drift


DATA_DRIFT = Gauge(
    "ml_data_drift_detected",
    "Whether any monitored feature drifted in the latest check.",
)
DRIFTED_FEATURES = Gauge(
    "ml_drifted_features_total",
    "Number of features that drifted in the latest check.",
)
CURRENT_ROWS = Gauge(
    "ml_current_rows",
    "Number of current-data rows in the latest check.",
)
FEATURE_DRIFT = Gauge(
    "ml_feature_drift_detected",
    "Whether an individual feature drifted.",
    ("feature", "test"),
)
FEATURE_P_VALUE = Gauge(
    "ml_feature_drift_p_value",
    "Statistical p-value for an individual feature.",
    ("feature", "test"),
)


def update_metrics(summary: dict[str, object]) -> None:
    DATA_DRIFT.set(int(bool(summary["drift_detected"])))
    DRIFTED_FEATURES.set(int(summary["drifted_features"]))
    CURRENT_ROWS.set(int(summary["current_rows"]))
    FEATURE_DRIFT.clear()
    FEATURE_P_VALUE.clear()
    for feature in summary["features"]:
        labels = (str(feature["feature"]), str(feature["test"]))
        FEATURE_DRIFT.labels(*labels).set(int(bool(feature["drift_detected"])))
        FEATURE_P_VALUE.labels(*labels).set(float(feature["p_value"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("data/reference.csv"))
    parser.add_argument("--current", type=Path, default=Path("data/current.csv"))
    parser.add_argument("--summary", type=Path, default=Path("drift_summary.json"))
    parser.add_argument("--report", type=Path, default=Path("drift_report.html"))
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    args = parser.parse_args()

    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be greater than zero")

    start_http_server(args.port)
    print(f"Prometheus metrics available at http://localhost:{args.port}/metrics")
    while True:
        result = detect_drift(
            args.reference,
            args.current,
            args.summary,
            args.report,
            args.threshold,
        )
        update_metrics(result)
        print(
            f"Drift check complete: {result['drifted_features']} drifted feature(s); "
            f"next check in {args.interval_seconds:g}s"
        )
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
