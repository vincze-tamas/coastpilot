#!/usr/bin/env python3
from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = Path(os.getenv("COASTPILOT_LABEL_INPUT", str(PROJECT_ROOT / "data" / "gold_istria_sample_ai_only.csv")))
OUTPUT_CSV = Path(os.getenv("COASTPILOT_LABEL_OUTPUT", str(PROJECT_ROOT / "data" / "gold_istria_sample_ai_only_labeled.csv")))
APP_TOKEN = os.getenv("COASTPILOT_LABEL_TOKEN", "")
LABEL_COLUMN = os.getenv("COASTPILOT_LABEL_COLUMN", "gold_label")
CONFIDENCE_COLUMN = os.getenv("COASTPILOT_CONFIDENCE_COLUMN", "gold_confidence")
NOTE_COLUMN = os.getenv("COASTPILOT_NOTE_COLUMN", "reviewer_note")
TIMESTAMP_COLUMN = os.getenv("COASTPILOT_TIMESTAMP_COLUMN", "labeled_at")

LABELS = ["rock", "pebble", "cobble", "sand", "built", "unknown"]
META_FIELDS = [
    ("error", "review_error_type"),
    ("gold", "gold_label"),
    ("AI", "ai_surface_class"),
    ("AI conf", "ai_confidence"),
    ("review label", "review_label"),
]

app = Flask(__name__)


def _check_token() -> None:
    if not APP_TOKEN:
        return
    token = request.args.get("token", "") or request.headers.get("X-Label-Token", "")
    if token != APP_TOKEN:
        abort(403)


def _required_output_columns() -> list[str]:
    return [LABEL_COLUMN, CONFIDENCE_COLUMN, NOTE_COLUMN, TIMESTAMP_COLUMN]


def _load_df() -> pd.DataFrame:
    if OUTPUT_CSV.exists():
        df = pd.read_csv(OUTPUT_CSV)
    else:
        df = pd.read_csv(INPUT_CSV)
        for c in _required_output_columns():
            if c not in df.columns:
                df[c] = ""
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
    for c in _required_output_columns():
        if c not in df.columns:
            df[c] = ""
        # force text dtype; prevents pandas float NaN columns from rejecting string labels
        df[c] = df[c].fillna("").astype(str)
    return df


def _save_df(df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_CSV, index=False)


def _first_unlabeled_index(df: pd.DataFrame) -> Optional[int]:
    m = df[LABEL_COLUMN].fillna("").astype(str).str.strip() == ""
    idx = df.index[m]
    if len(idx) == 0:
        return None
    return int(idx[0])


@app.route("/image")
def image_proxy():
    _check_token()
    idx = request.args.get("idx", type=int)
    if idx is None:
        abort(400)
    df = _load_df()
    if idx < 0 or idx >= len(df):
        abort(404)

    image_path = str(df.iloc[idx].get("vision_image_path", "") or "").strip()
    if image_path:
        p = Path(image_path)
        if not p.is_absolute():
            p = (PROJECT_ROOT / image_path).resolve()
        if p.exists() and p.is_file():
            return send_file(p)
    abort(404)


PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Coastpilot Review Labeler</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 16px; background: #0f172a; color: #e2e8f0; }
    .wrap { max-width: 1180px; margin: 0 auto; }
    .row { display: flex; gap: 20px; flex-wrap: wrap; }
    .card { background: #111827; border-radius: 14px; padding: 14px; }
    .imgbox { flex: 2; min-width: 320px; }
    .ctrl { flex: 1; min-width: 300px; }
    img { width: 100%; height: auto; border-radius: 10px; background: #1f2937; }
    button { width: 100%; padding: 14px; margin: 6px 0; border: 0; border-radius: 10px; font-size: 18px; font-weight: 700; cursor: pointer; }
    .rock { background: #f59e0b; } .pebble { background: #06b6d4; } .cobble { background: #8b5cf6; }
    .sand { background: #eab308; } .built { background: #22c55e; } .unknown { background: #6b7280; color: #fff; }
    .meta { font-size: 14px; line-height: 1.5; color: #cbd5e1; }
    .pill { display: inline-block; padding: 3px 8px; border-radius: 999px; background: #334155; color: #fff; font-weight: 700; }
    .warn { color: #fbbf24; }
    input, textarea { width: 100%; margin-top: 8px; border-radius: 8px; border: 1px solid #374151; background: #0b1220; color: #e2e8f0; padding: 8px; box-sizing: border-box; }
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Coastpilot Error Review</h2>
    {% if done %}
      <div class="card"><h3>Kész. Minden sor review-zva.</h3></div>
    {% else %}
    <div class="row">
      <div class="card imgbox">
        {% if image_ok %}
          <img src="{{ image_src }}" alt="segment image" />
        {% else %}
          <div style="padding:20px;background:#1f2937;border-radius:10px">Nincs helyi kép. URL: <a href="{{ vision_image_url }}" target="_blank" style="color:#93c5fd">{{ vision_image_url }}</a></div>
        {% endif %}
      </div>
      <div class="card ctrl">
        <div class="meta">
          <div><b>Progress:</b> {{ done_count }}/{{ total }} ({{ progress }}%)</div>
          <div><b>Queue row:</b> {{ idx + 1 }}</div>
          <div><b>segment_id:</b> <span class="pill">{{ segment_id }}</span></div>
          {% for label, value in meta %}
            <div><b>{{ label }}:</b> <span class="warn">{{ value }}</span></div>
          {% endfor %}
          <div><b>Writing to:</b> {{ label_column }}</div>
        </div>
        <form method="post" action="/label{% if token %}?token={{ token }}{% endif %}">
          <input type="hidden" name="idx" value="{{ idx }}" />
          <input name="confidence" placeholder="confidence: low/medium/high (optional)" />
          <textarea name="note" rows="2" placeholder="note: why changed / disputed / unclear (optional)"></textarea>
          {% for l in labels %}
            <button class="{{ l }}" name="label" value="{{ l }}">{{ l|upper }}</button>
          {% endfor %}
        </form>
      </div>
    </div>
    {% endif %}
  </div>
</body>
</html>
"""


@app.route("/")
def index():
    _check_token()
    df = _load_df()
    idx = _first_unlabeled_index(df)
    total = len(df)
    done_count = int((df[LABEL_COLUMN].fillna("").astype(str).str.strip() != "").sum())
    progress = round((done_count / total) * 100, 1) if total else 100.0

    if idx is None:
        return render_template_string(PAGE, done=True)

    row = df.iloc[idx]
    token = request.args.get("token", "")
    image_path = str(row.get("vision_image_path", "") or "").strip()
    image_ok = False
    if image_path:
        p = Path(image_path)
        if not p.is_absolute():
            p = (PROJECT_ROOT / image_path).resolve()
        image_ok = p.exists() and p.is_file()

    image_src = url_for("image_proxy", idx=idx, token=token) if image_ok else ""
    meta = [(label, row.get(col, "")) for label, col in META_FIELDS if col in df.columns]

    return render_template_string(
        PAGE,
        done=False,
        labels=LABELS,
        idx=idx,
        total=total,
        done_count=done_count,
        progress=progress,
        segment_id=row.get("segment_id", ""),
        meta=meta,
        vision_image_url=row.get("vision_image_url", ""),
        image_ok=image_ok,
        image_src=image_src,
        token=token,
        label_column=LABEL_COLUMN,
    )


@app.route("/label", methods=["POST"])
def label():
    _check_token()
    label_value = (request.form.get("label") or "").strip().lower()
    if label_value not in LABELS:
        abort(400)
    idx = request.form.get("idx", type=int)
    if idx is None:
        abort(400)

    df = _load_df()
    if idx < 0 or idx >= len(df):
        abort(404)

    df.at[idx, LABEL_COLUMN] = label_value
    df.at[idx, CONFIDENCE_COLUMN] = (request.form.get("confidence") or "").strip()
    df.at[idx, NOTE_COLUMN] = (request.form.get("note") or "").strip()
    df.at[idx, TIMESTAMP_COLUMN] = pd.Timestamp.utcnow().isoformat()
    _save_df(df)

    token = request.args.get("token", "")
    if token:
        return redirect(url_for("index", token=token))
    return redirect(url_for("index"))


def _validate_bind_host(host: str) -> None:
    """Require authentication before exposing the standalone labeler off-loopback."""
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.lower() == "localhost"
    if not loopback and not APP_TOKEN.strip():
        raise SystemExit(
            "COASTPILOT_LABEL_TOKEN is required for a non-loopback COASTPILOT_LABEL_HOST"
        )


if __name__ == "__main__":
    host = os.getenv("COASTPILOT_LABEL_HOST", "127.0.0.1").strip()
    _validate_bind_host(host)
    port = int(os.getenv("COASTPILOT_LABEL_PORT", "8787"))
    app.run(host=host, port=port, debug=False)
