# Istria productization plan

Goal
Help a rock-loving traveler quickly find coast segments explicitly labeled rock.

Implemented now
- Gold dataset sampling script: scripts/build_gold_sample.py
- Gold evaluator script: scripts/evaluate_gold.py
- Rock-oriented recommendation score added to pipeline output (recommendation_score, recommendation_band)
- Rock confidence threshold control (--rock-min-confidence)
- Map UI generator: scripts/build_map_ui.py -> public/coastpilot_map.html
- Automatic quality report generation in pipeline run (data/quality_latest.json by default)

Workflow
1. Generate or refresh classified data.
2. Build annotation sample (400 rows by default).
3. Manually annotate gold_label in CSV.
4. Run evaluator and inspect strict_rock precision/recall/f1. Only gold_label == rock is positive; pebble/cobble are negatives.
5. Tune model prompt thresholds and rerun.
6. Rebuild map HTML and use class confidence score filters.

Commands
python scripts/build_gold_sample.py --input data/segments_classified.geojson --output data/gold_istria_sample.csv --size 400
python scripts/evaluate_gold.py --gold-csv data/gold_istria_sample.csv --output-json data/gold_eval_latest.json
python run_pipeline.py --step ai --vision-limit 100 --rock-min-confidence 0.60 --prefer-rock-weight 1.0
python scripts/build_map_ui.py --input data/segments_classified.geojson --output public/coastpilot_map.html

Next practical target
- Reach strict_rock.f1 >= 0.80 on manually annotated Istria sample before scaling to other regions.

Label policy
- Canonical labels: rock, pebble, cobble, sand, built, unknown.
- Strict rocky: positive iff label == rock. Pebble and cobble are NOT rocky.
- Evaluator rejects invalid gold labels and reports invalid prediction labels separately.
