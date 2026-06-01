#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description='Evaluate model labels against gold annotations')
    p.add_argument('--gold-csv', required=True)
    p.add_argument('--output-json', required=True)
    args = p.parse_args()

    df = pd.read_csv(args.gold_csv)
    df = df[df['gold_label'].notna() & (df['gold_label'].astype(str).str.strip() != '')].copy()
    if df.empty:
        raise SystemExit('No annotated gold_label rows found')

    df['pred'] = df['ai_surface_class'].fillna('unknown').astype(str).str.strip().str.lower()
    df['gold'] = df['gold_label'].fillna('unknown').astype(str).str.strip().str.lower()

    acc = float((df['pred'] == df['gold']).mean())
    tp = int(((df['pred'] == 'rock') & (df['gold'] == 'rock')).sum())
    fp = int(((df['pred'] == 'rock') & (df['gold'] != 'rock')).sum())
    fn = int(((df['pred'] != 'rock') & (df['gold'] == 'rock')).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    out = {
        'rows': int(len(df)),
        'accuracy': round(acc, 4),
        'rock_precision': round(precision, 4),
        'rock_recall': round(recall, 4),
        'rock_f1': round(f1, 4),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
