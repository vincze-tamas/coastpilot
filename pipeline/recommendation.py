from __future__ import annotations


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def preference_score(label: str | None, confidence: float | None, *, prefer_rock: float = 1.0) -> float:
    base = {
        'rock': 1.0,
        'pebble': 0.65,
        'built': 0.35,
        'sand': 0.2,
        'unknown': 0.0,
        None: 0.0,
    }.get(label, 0.0)
    conf = 0.5 if confidence is None else clamp(float(confidence))
    pref = clamp(prefer_rock)
    if label == 'rock':
        score = base * (0.6 + 0.4 * pref) * conf
    else:
        score = base * (1.0 - 0.5 * pref) * conf
    return round(clamp(score), 4)


def recommendation_band(score: float) -> str:
    if score >= 0.75:
        return 'strong_match'
    if score >= 0.45:
        return 'possible_match'
    return 'weak_match'
