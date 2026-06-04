#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

CANONICAL_LABELS = ("rock", "pebble", "cobble", "sand", "built", "unknown")
VALID_LABELS = set(CANONICAL_LABELS)
STRICT_ROCK_DEFINITION = "positive iff label == rock (pebble/cobble are NOT rocky)"
PREDICTION_COLUMNS = ("ai_surface_class", "ai_confidence", "ai_provider", "ai_vision_version")


def norm(v: object) -> str:
    if pd.isna(v):
        return "unknown"
    s = str(v).strip().lower()
    return s if s else "unknown"


def bin_rock(v: str) -> int:
    # strict rocky: ONLY rock is positive
    return 1 if v == "rock" else 0


def load_predictions_geojson(path: str) -> pd.DataFrame:
    preds = gpd.read_file(path)
    if "segment_id" not in preds.columns:
        raise SystemExit(f"Predictions file {path} has no segment_id column")
    keep = [c for c in ("segment_id", *PREDICTION_COLUMNS) if c in preds.columns]
    preds = pd.DataFrame(preds[keep]).copy()
    preds["segment_id"] = preds["segment_id"].astype(str)
    dup = preds[preds["segment_id"].duplicated()]["segment_id"].unique().tolist()
    if dup:
        raise SystemExit(f"Predictions file has duplicate segment_id values: {dup[:10]}")
    rename = {c: f"pred_{c}" for c in PREDICTION_COLUMNS if c in preds.columns}
    return preds.rename(columns=rename)


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate model labels against gold annotations")
    p.add_argument("--gold-csv", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--predictions-geojson", default=None, help="Optional predictions source; joined to gold rows by exact segment_id")
    p.add_argument("--fail-on-missing-predictions", action="store_true", help="Fail if any gold segment_id has no usable prediction")
    args = p.parse_args()

    df = pd.read_csv(args.gold_csv)
    df = df[df["gold_label"].notna() & (df["gold_label"].astype(str).str.strip() != "")].copy()
    if df.empty:
        raise SystemExit("No annotated gold_label rows found")
    if "segment_id" not in df.columns:
        raise SystemExit("Gold CSV has no segment_id column")
    df["segment_id"] = df["segment_id"].astype(str)
    duplicate_gold = df[df["segment_id"].duplicated()]["segment_id"].unique().tolist()
    if duplicate_gold:
        raise SystemExit(f"Gold CSV has duplicate segment_id values: {duplicate_gold[:10]}")

    prediction_source = "gold_csv.ai_surface_class"
    if args.predictions_geojson:
        preds = load_predictions_geojson(args.predictions_geojson)
        df = df.drop(columns=[c for c in PREDICTION_COLUMNS if c in df.columns], errors="ignore")
        df = df.merge(preds, on="segment_id", how="left")
        for c in PREDICTION_COLUMNS:
            pred_col = f"pred_{c}"
            if pred_col in df.columns:
                df[c] = df[pred_col]
        prediction_source = args.predictions_geojson

    if "ai_surface_class" not in df.columns:
        raise SystemExit("No ai_surface_class predictions available; pass --predictions-geojson or include ai_surface_class in gold CSV")

    df["pred"] = df["ai_surface_class"].apply(norm)
    df["gold"] = df["gold_label"].apply(norm)

    bad_gold = sorted(set(df[~df["gold"].isin(VALID_LABELS)]["gold"].tolist()))
    if bad_gold:
        raise SystemExit(f"Invalid gold labels found: {bad_gold}. Allowed: {sorted(VALID_LABELS)}")

    bad_pred = sorted(set(df[~df["pred"].isin(VALID_LABELS)]["pred"].tolist()))

    prediction_present = df["ai_surface_class"].notna() & (df["ai_surface_class"].astype(str).str.strip() != "") & (df["pred"] != "unknown")
    missing_prediction_ids = df.loc[~prediction_present, "segment_id"].tolist()
    if args.fail_on_missing_predictions and missing_prediction_ids:
        raise SystemExit(
            f"Missing usable predictions for {len(missing_prediction_ids)}/{len(df)} gold segment_ids; "
            f"first ids: {missing_prediction_ids[:20]}"
        )

    acc = float((df["pred"] == df["gold"]).mean())

    pred_rock = df["pred"].apply(bin_rock)
    gold_rock = df["gold"].apply(bin_rock)

    tp = int(((pred_rock == 1) & (gold_rock == 1)).sum())
    fp = int(((pred_rock == 1) & (gold_rock == 0)).sum())
    fn = int(((pred_rock == 0) & (gold_rock == 1)).sum())
    tn = int(((pred_rock == 0) & (gold_rock == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    out = {
        "rows": int(len(df)),
        "accuracy": round(acc, 4),
        "prediction_source": prediction_source,
        "prediction_coverage": {
            "usable_predictions": int(prediction_present.sum()),
            "missing_or_unknown_predictions": int((~prediction_present).sum()),
            "first_missing_segment_ids": missing_prediction_ids[:20],
        },
        "label_counts": {
            "gold": df["gold"].value_counts().to_dict(),
            "pred": df["pred"].value_counts().to_dict(),
        },
        "strict_rock": {
            "definition": STRICT_ROCK_DEFINITION,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        # Backward-compatible top-level aliases; strict_rock is canonical.
        "rock_precision": round(precision, 4),
        "rock_recall": round(recall, 4),
        "rock_f1": round(f1, 4),
        "invalid_pred_labels": bad_pred,
        "allowed_labels": list(CANONICAL_LABELS),
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
