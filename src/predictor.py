"""
Prediction Logic Module
=======================
Takes a trained model and live feature data → outputs actionable predictions.

Outputs:
  1. Risk Probability (0–1)
  2. Risk Level (Green / Yellow / Red)
  3. Estimated time-to-congestion (minutes)
  4. Signage message (if high risk)

Signage trigger logic:
  - prob > 0.7 → Trigger zone-specific redirect message
  - prob > 0.85 → Trigger emergency-level message
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

try:
    from src.model import load_model
    from src.features import get_feature_columns, get_realtime_features, PREDICTION_HORIZON
    from src.aws_bedrock import generate_signage_message as bedrock_signage, is_bedrock_available
    from src.aws_storage import store_prediction, is_dynamodb_available
except ImportError:
    from model import load_model
    from features import get_feature_columns, get_realtime_features, PREDICTION_HORIZON
    try:
        from aws_bedrock import generate_signage_message as bedrock_signage, is_bedrock_available
        from aws_storage import store_prediction, is_dynamodb_available
    except ImportError:
        bedrock_signage = None
        is_bedrock_available = lambda: False
        store_prediction = lambda **kwargs: False
        is_dynamodb_available = lambda: False

logger = logging.getLogger(__name__)

# ── Bedrock usage flag (can be toggled from dashboard) ──
_use_bedrock = True


def set_use_bedrock(enabled: bool):
    """Toggle Bedrock-powered signage on/off."""
    global _use_bedrock
    _use_bedrock = enabled


def get_use_bedrock() -> bool:
    """Check if Bedrock signage is enabled."""
    return _use_bedrock and is_bedrock_available()


# ── Signage messages per zone ──
SIGNAGE_MESSAGES = {
    "Zone_A": {
        "warning": "⚠️ Zone A: Crowd building. Please consider Gate B.",
        "critical": "🚨 Zone A RESTRICTED — Redirect to Gate B immediately!",
        "emergency": "🆘 EMERGENCY: Zone A blocked. Use Alternate Exit NOW!",
    },
    "Zone_B": {
        "warning": "⚠️ Zone B: Moderate congestion ahead. Use side corridor.",
        "critical": "🚨 Zone B congested — Redirect to Zone C entrance!",
        "emergency": "🆘 EMERGENCY: Zone B blocked. Follow evacuation signs!",
    },
    "Zone_C": {
        "warning": "⚠️ Zone C: Increasing crowd density. Allow extra time.",
        "critical": "🚨 Zone C approaching capacity — Use Alternate Exit!",
        "emergency": "🆘 EMERGENCY: All zones critical. Follow staff directions!",
    },
}

# Risk thresholds
YELLOW_THRESHOLD = 0.4
RED_THRESHOLD = 0.7
EMERGENCY_THRESHOLD = 0.85


@dataclass
class PredictionResult:
    """Structured prediction output for one zone."""
    zone_id: str
    risk_probability: float
    risk_level: str         # "green", "yellow", "red"
    risk_color: str         # hex color for UI
    time_to_congestion: float  # minutes, -1 if no risk
    signage_message: str    # empty if no action needed
    signage_active: bool


def get_risk_level(prob: float) -> tuple[str, str]:
    """Map probability to risk level and color."""
    if prob >= RED_THRESHOLD:
        return "red", "#FF4444"
    elif prob >= YELLOW_THRESHOLD:
        return "yellow", "#FFAA00"
    else:
        return "green", "#44BB44"


def estimate_time_to_congestion(
    prob: float,
    density_rate: float,
    current_density: float,
    interval_seconds: int = 30,
) -> float:
    """
    Estimate minutes until congestion based on:
    - Current risk probability
    - Rate of density change
    - Distance from density threshold

    Returns -1 if no congestion is expected.
    """
    if prob < YELLOW_THRESHOLD:
        return -1.0

    density_threshold = 4.0  # Same as in features.py

    if density_rate <= 0.01:
        # Density not increasing — can't estimate time
        if prob >= RED_THRESHOLD:
            return 2.0  # Already near/at congestion
        return -1.0

    # Steps until density reaches threshold
    density_gap = max(density_threshold - current_density, 0.1)
    steps_to_threshold = density_gap / max(density_rate, 0.01)
    minutes_to_congestion = (steps_to_threshold * interval_seconds) / 60.0

    # Clamp to reasonable range
    minutes_to_congestion = max(0.5, min(minutes_to_congestion, 30.0))

    # If already high risk, override with short time
    if prob >= EMERGENCY_THRESHOLD:
        minutes_to_congestion = min(minutes_to_congestion, 2.0)
    elif prob >= RED_THRESHOLD:
        minutes_to_congestion = min(minutes_to_congestion, 8.0)

    return round(minutes_to_congestion, 1)


def get_signage_message(
    zone_id: str,
    prob: float,
    density: float = 0.0,
    velocity: float = 0.0,
    time_to_congestion: float = -1.0,
) -> tuple[str, bool, bool]:
    """
    Get the appropriate digital signage message for a zone based on risk.
    Tries Amazon Bedrock first for AI-generated messages, falls back to static templates.

    Returns (message, is_active, used_bedrock).
    """
    # Determine risk level for Bedrock prompt
    if prob >= EMERGENCY_THRESHOLD:
        risk_label = "emergency"
    elif prob >= RED_THRESHOLD:
        risk_label = "critical"
    elif prob >= YELLOW_THRESHOLD:
        risk_label = "warning"
    else:
        return "✅ Normal flow. No action required.", False, False

    # ── Try Bedrock AI-generated message ──
    used_bedrock = False
    if get_use_bedrock() and bedrock_signage is not None:
        try:
            ai_message = bedrock_signage(
                zone_id=zone_id,
                risk_level=risk_label,
                risk_probability=prob,
                density=density,
                velocity=velocity,
                time_to_congestion=time_to_congestion,
            )
            if ai_message:
                # Add appropriate emoji prefix
                if risk_label == "emergency":
                    prefix = "🆘 "
                elif risk_label == "critical":
                    prefix = "🚨 "
                else:
                    prefix = "⚠️ "
                used_bedrock = True
                return f"{prefix}{ai_message}", True, True
        except Exception as e:
            logger.warning(f"Bedrock signage failed, using fallback: {e}")

    # ── Fallback: static template messages ──
    zone_messages = SIGNAGE_MESSAGES.get(zone_id, SIGNAGE_MESSAGES["Zone_A"])
    return zone_messages[risk_label], True, False


def predict_zone(
    zone_id: str,
    features: dict,
    model=None,
    scaler=None,
) -> PredictionResult:
    """
    Make a full prediction for one zone given its current features.
    Returns a structured PredictionResult.
    """
    if model is None or scaler is None:
        model, scaler = load_model()

    feature_cols = get_feature_columns()
    feature_vector = np.array([[features.get(col, 0) for col in feature_cols]])

    # Scale features and predict probability
    feature_scaled = scaler.transform(feature_vector)
    prob = model.predict_proba(feature_scaled)[0][1]

    # Derive all outputs
    risk_level, risk_color = get_risk_level(prob)
    time_to_cong = estimate_time_to_congestion(
        prob=prob,
        density_rate=features.get("density_rate_of_change", 0),
        current_density=features.get("density", 0),
    )
    message, signage_active, used_bedrock = get_signage_message(
        zone_id=zone_id,
        prob=prob,
        density=features.get("density", 0),
        velocity=features.get("velocity", 0),
        time_to_congestion=time_to_cong,
    )

    result = PredictionResult(
        zone_id=zone_id,
        risk_probability=round(prob, 4),
        risk_level=risk_level,
        risk_color=risk_color,
        time_to_congestion=time_to_cong,
        signage_message=message,
        signage_active=signage_active,
    )

    # ── Log prediction to DynamoDB (async-safe, non-blocking) ──
    if is_dynamodb_available():
        try:
            store_prediction(
                zone_id=zone_id,
                timestamp=datetime.utcnow().isoformat(),
                risk_probability=prob,
                risk_level=risk_level,
                density=features.get("density", 0),
                velocity=features.get("velocity", 0),
                time_to_congestion=time_to_cong,
                signage_message=message,
            )
        except Exception as e:
            logger.debug(f"DynamoDB logging skipped: {e}")

    return result


def predict_all_zones(
    zone_histories: dict[str, pd.DataFrame],
    model=None,
    scaler=None,
) -> dict[str, PredictionResult]:
    """
    Predict congestion risk for all zones given their recent history.
    zone_histories: {zone_id: DataFrame of recent readings}
    """
    if model is None or scaler is None:
        model, scaler = load_model()

    results = {}
    for zone_id, history_df in zone_histories.items():
        features = get_realtime_features(history_df)
        if features:
            results[zone_id] = predict_zone(zone_id, features, model, scaler)

    return results


if __name__ == "__main__":
    # Quick test with synthetic features
    model, scaler = load_model()

    # Simulate a normal situation
    normal_features = {
        "density": 1.2,
        "velocity": 1.5,
        "rolling_density_mean": 1.1,
        "rolling_velocity_mean": 1.5,
        "density_rate_of_change": 0.02,
        "velocity_rate_of_change": -0.01,
        "density_velocity_ratio": 0.8,
    }

    # Simulate a pre-congestion situation
    risky_features = {
        "density": 3.8,
        "velocity": 0.6,
        "rolling_density_mean": 3.2,
        "rolling_velocity_mean": 0.8,
        "density_rate_of_change": 0.35,
        "velocity_rate_of_change": -0.15,
        "density_velocity_ratio": 6.3,
    }

    print("── Normal Situation ──")
    result = predict_zone("Zone_A", normal_features, model, scaler)
    print(f"   Risk: {result.risk_probability:.1%} ({result.risk_level})")
    print(f"   Signage: {result.signage_message}")

    print("\n── Pre-Congestion Situation ──")
    result = predict_zone("Zone_A", risky_features, model, scaler)
    print(f"   Risk: {result.risk_probability:.1%} ({result.risk_level})")
    print(f"   Time to congestion: {result.time_to_congestion} min")
    print(f"   Signage: {result.signage_message}")
