"""
predictor.py
============
Real-time prediction logic for CrowdAI.

Responsibilities:
  - Take engineered features for a zone and run them through the trained model
  - Compute risk level, risk colour, time-to-congestion estimate
  - Generate digital signage messages (static templates or Amazon Bedrock)
  - Return a PredictionResult dataclass consumed by app.py

Risk thresholds (matching README):
  Green  — risk probability  < 40%
  Yellow — risk probability 40% – 70%
  Red    — risk probability  > 70%
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ── Thresholds ────────────────────────────────────────────────────────────────
YELLOW_THRESHOLD = 0.40   # 40%
RED_THRESHOLD    = 0.70   # 70%

# ── Bedrock toggle (module-level state) ───────────────────────────────────────
_use_bedrock: bool = False

def set_use_bedrock(enabled: bool) -> None:
    global _use_bedrock
    _use_bedrock = enabled

def get_use_bedrock() -> bool:
    return _use_bedrock


# ── PredictionResult ──────────────────────────────────────────────────────────
@dataclass
class PredictionResult:
    """
    All outputs for a single zone at a single simulation step.

    Fields used by app.py
    ---------------------
    zone_id            : e.g. "Zone A"
    risk_probability   : float 0-1
    risk_level         : "green" | "yellow" | "red"
    risk_color         : hex string for UI colouring
    time_to_congestion : estimated minutes until congestion (0 = no risk)
    signage_active     : True when risk >= RED_THRESHOLD
    signage_message    : human-readable text for the digital sign
    density            : latest raw density reading  (p/m²)  ← used by detail view
    velocity           : latest raw velocity reading (m/s)   ← used by detail view
    grid               : 2-D heatmap grid — filled by app.py after predict_zone()
    """
    zone_id            : str
    risk_probability   : float
    risk_level         : str                          # "green" | "yellow" | "red"
    risk_color         : str                          # hex colour
    time_to_congestion : int                          # minutes, 0 = no risk
    signage_active     : bool
    signage_message    : str
    density            : float = 0.0                 # raw density  (p/m²)
    velocity           : float = 0.0                 # raw velocity (m/s)
    grid               : list  = field(default_factory=list)  # heatmap grid (set in app.py)


# ── Risk helpers ──────────────────────────────────────────────────────────────
def _risk_level(prob: float) -> str:
    if prob >= RED_THRESHOLD:
        return "red"
    if prob >= YELLOW_THRESHOLD:
        return "yellow"
    return "green"


def _risk_color(level: str) -> str:
    return {
        "red":    "#EF4444",
        "yellow": "#F59E0B",
        "green":  "#10B981",
    }.get(level, "#10B981")


def _time_to_congestion(prob: float, level: str) -> int:
    """
    Estimate minutes until congestion based on risk probability.

    Logic mirrors the README description of 10-15 min prediction window:
      Red    →  2–8  min  (imminent)
      Yellow →  9–15 min  (building)
      Green  →  0        (no risk)
    """
    if level == "red":
        # Higher probability = less time remaining
        base = max(1, int((1.0 - prob) * 20))
        return min(base, 8)
    if level == "yellow":
        base = max(9, int((1.0 - prob) * 30))
        return min(base, 15)
    
    # Green - no imminent risk, high baseline
    base = int(60 - prob * 100)
    return max(15, min(base, 60))


# ── Static signage templates ──────────────────────────────────────────────────
_SIGNAGE_GREEN = [
    "✅ {zone} is clear — enjoy the event!",
    "✅ Normal flow in {zone}. No action needed.",
    "✅ {zone} operating normally. All clear.",
]

_SIGNAGE_YELLOW = [
    "⚡ {zone} getting busy — consider using an alternate route.",
    "⚡ Elevated crowd levels in {zone}. Stewards on standby.",
    "⚡ {zone} approaching capacity. Please spread out.",
    "⚡ Crowd building in {zone} — ~{ttc} min to peak. Use Gate B as alternate.",
]

_SIGNAGE_RED = [
    "⚠️ {zone} CRITICAL — congestion expected in ~{ttc} min. Please use alternate exits.",
    "🚨 URGENT: {zone} at capacity. Crowd control deployed. Use alternate route NOW.",
    "🚨 {zone} RED ALERT — redirect immediately. Congestion in ~{ttc} min.",
    "⚠️ CROWDAI ALERT: {zone} critical. All stewards to {zone} immediately.",
]


def _static_signage(zone_id: str, level: str, ttc: int) -> str:
    templates = {
        "green":  _SIGNAGE_GREEN,
        "yellow": _SIGNAGE_YELLOW,
        "red":    _SIGNAGE_RED,
    }.get(level, _SIGNAGE_GREEN)

    template = random.choice(templates)
    return template.format(zone=zone_id, ttc=ttc)


# ── Bedrock signage wrapper ───────────────────────────────────────────────────
def _bedrock_signage(
    zone_id: str,
    level: str,
    ttc: int,
    density: float,
    velocity: float,
) -> str | None:
    """
    Attempt to generate a signage message via Amazon Bedrock.
    Returns None if Bedrock is unavailable or raises an exception.
    """
    try:
        from src.aws_bedrock import generate_crowd_recommendation, is_bedrock_available
        if not is_bedrock_available():
            return None
        zone_data = {
            "risk_level":         level,
            "risk_probability":   RED_THRESHOLD if level == "red" else YELLOW_THRESHOLD if level == "yellow" else 0.1,
            "density":            density,
            "velocity":           velocity,
            "time_to_congestion": ttc,
        }
        return generate_crowd_recommendation(zone_id, zone_data)
    except Exception:
        return None


# ── Core prediction function ──────────────────────────────────────────────────
def predict_zone(
    zone_id:  str,
    features: dict[str, Any],
    model:    Any,
    scaler:   Any,
) -> PredictionResult:
    """
    Run the trained model for one zone and return a PredictionResult.

    Parameters
    ----------
    zone_id  : zone identifier, e.g. "Zone A"
    features : dict produced by src.features.get_realtime_features()
               Expected keys (7 features, matching model training):
                 - rolling_density_mean
                 - rolling_velocity_mean
                 - density_rate_of_change
                 - velocity_rate_of_change
                 - density_velocity_ratio
                 - density
                 - velocity
    model    : trained sklearn estimator (LogisticRegression)
    scaler   : fitted sklearn scaler (StandardScaler)

    Returns
    -------
    PredictionResult
    """
    from src.features import get_feature_columns

    # ── Build feature vector in the exact column order used at training ──
    feature_cols = get_feature_columns()
    X_raw = np.array([[features[col] for col in feature_cols]])

    # ── Scale ──
    X_scaled = scaler.transform(X_raw)

    # ── Predict ──
    prob  = float(model.predict_proba(X_scaled)[0][1])   # probability of congestion
    level = _risk_level(prob)
    color = _risk_color(level)
    ttc   = _time_to_congestion(prob, level)

    # ── Raw sensor values (for KPI tiles and detail view) ──
    density  = float(features.get("density",  0.0))
    velocity = float(features.get("velocity", 0.0))

    # ── Signage message ──
    signage_active = level == "red"

    if _use_bedrock and signage_active:
        message = _bedrock_signage(zone_id, level, ttc, density, velocity) \
                  or _static_signage(zone_id, level, ttc)
    else:
        message = _static_signage(zone_id, level, ttc)

    return PredictionResult(
        zone_id            = zone_id,
        risk_probability   = prob,
        risk_level         = level,
        risk_color         = color,
        time_to_congestion = ttc,
        signage_active     = signage_active,
        signage_message    = message,
        density            = density,
        velocity           = velocity,
        grid               = [],        # populated by app.py after this call
    )