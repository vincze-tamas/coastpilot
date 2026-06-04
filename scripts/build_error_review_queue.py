#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

PREDICTION_COLUMNS = ("ai_surface_class", "ai_confidence", "ai_provider", "ai_vision_version")
REVIEW_COLUMNS = ("review_label", "review_confidence", "review_note", "reviewed_at")


def norm(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    return text if text else "unknown"


def load_predictions(predictions_geojson: str | Path) -> pd.DataFrame:
    preds = gpd.read_file(predictions_geojson)
    if "segment_id" not in preds.columns:
        raise SystemExit(f"Predictions file {predictions_geojson} has no segment_id column")
    keep = [c for c in ("segment_id", *PREDICTION_COLUMNS) if c in preds.columns]
    preds = pd.DataFrame(preds[keep]).copy()
    preds["segment_id"] = preds["segment_id"].astype(str)
    dup = preds[preds["segment_id"].duplicated()]["segment_id"].unique().tolist()
    if dup:
        raise SystemExit(f"Predictions file has duplicate segment_id values: {dup[:10]}")
    return preds


def build_queue(gold_csv: str | Path, predictions_geojson: str | Path, output_csv: str | Path) -> dict:
    gold = pd.read_csv(gold_csv)
    if "segment_id" not in gold.columns:
        raise SystemExit("Gold CSV has no segment_id column")
    if "gold_label" not in gold.columns:
        raise SystemExit("Gold CSV has no gold_label column")

    gold = gold[gold["gold_label"].notna() & (gold["gold_label"].astype(str).str.strip() != "")].copy()
    gold["segment_id"] = gold["segment_id"].astype(str)
    gold = gold.drop(columns=[c for c in PREDICTION_COLUMNS if c in gold.columns], errors="ignore")

    preds = load_predictions(predictions_geojson)
    df = gold.merge(preds, on="segment_id", how="left")
    df["gold_norm"] = df["gold_label"].map(norm)
    df["pred_norm"] = df["ai_surface_class"].map(norm) if "ai_surface_class" in df.columns else "unknown"

    prediction_present = df["ai_surface_class"].notna() & (df["ai_surface_class"].astype(str).str.strip() != "") & (df["pred_norm"] != "unknown")
    false_positive = prediction_present & df["pred_norm"].eq("rock") & ~df["gold_norm"].eq("rock")
    false_negative = prediction_present & ~df["pred_norm"].eq("rock") & df["gold_norm"].eq("rock")
    missing = ~prediction_present

    df.loc[false_positive, "review_error_type"] = "false_positive_rock"
    df.loc[false_negative, "review_error_type"] = "false_negative_rock"
    df.loc[missing, "review_error_type"] = "missing_prediction"

    queue = df[false_positive | false_negative | missing].copy()
    priority = {"false_positive_rock": 0, "false_negative_rock": 1, "missing_prediction": 2}
    queue["_priority"] = queue["review_error_type"].map(priority).fillna(99).astype(int)
    if "ai_confidence" not in queue.columns:
        queue["ai_confidence"] = ""
    queue["_conf_sort"] = pd.to_numeric(queue["ai_confidence"], errors="coerce").fillna(-1)
    queue = queue.sort_values(["_priority", "_conf_sort", "segment_id"], ascending=[True, False, True])

    for col in REVIEW_COLUMNS:
        if col not in queue.columns:
            queue[col] = ""

    front = [
        "segment_id", "review_error_type", "gold_label", "ai_surface_class", "ai_confidence",
        "vision_image_url", "vision_image_path", *REVIEW_COLUMNS,
    ]
    excluded = {"gold_norm", "pred_norm", "_priority", "_conf_sort"}
    cols = [c for c in front if c in queue.columns] + [c for c in queue.columns if c not in set(front) | excluded]
    queue = queue[cols]

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(output_csv, index=False)

    return {
        "rows": int(len(queue)),
        "false_positive_rock": int(false_positive.sum()),
        "false_negative_rock": int(false_negative.sum()),
        "missing_prediction": int(missing.sum()),
        "output_csv": str(output_csv),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Build Coastpilot strict-rock error review queue CSV")
    p.add_argument("--gold-csv", default="data/gold_istria_sample.csv")
    p.add_argument("--predictions-geojson", default="data/segments_classified.geojson")
    p.add_argument("--output-csv", default="data/error_review_queue.csv")
    p.add_argument("--summary-json", default="data/error_review_queue_summary.json")
    args = p.parse_args()

    summary = build_queue(args.gold_csv, args.predictions_geojson, args.output_csv)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
