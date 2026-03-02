"""
AWS Lambda Handler
==================
Serverless prediction endpoint for CrowdAI.

Deployed as an AWS Lambda function behind Amazon API Gateway.
Loads model artifacts from S3 at cold start, then processes
prediction requests with sub-100ms latency on warm invocations.

API Gateway Integration:
  POST /predict
  Body: {
    "zone_id": "Zone_A",
    "features": {
      "density": 3.8,
      "velocity": 0.6,
      "rolling_density_mean": 3.2,
      "rolling_velocity_mean": 0.8,
      "density_rate_of_change": 0.35,
      "velocity_rate_of_change": -0.15,
      "density_velocity_ratio": 6.3
    }
  }

  Response: {
    "zone_id": "Zone_A",
    "risk_probability": 0.82,
    "risk_level": "red",
    "risk_color": "#FF4444",
    "time_to_congestion": 5.2,
    "signage_message": "🚨 Zone A congested — Redirect to Gate B immediately!",
    "signage_active": true
  }
"""

import json
import os
import pickle
import logging
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── S3 configuration ──
S3_BUCKET = os.environ.get("CROWDAI_S3_BUCKET", "crowdai-models")
MODEL_KEY = "congestion_model.pkl"
SCALER_KEY = "scaler.pkl"

# ── Risk thresholds (same as predictor.py) ──
YELLOW_THRESHOLD = 0.4
RED_THRESHOLD = 0.7
EMERGENCY_THRESHOLD = 0.85

# ── Feature columns (same order as model.py) ──
FEATURE_COLUMNS = [
    "density",
    "velocity",
    "rolling_density_mean",
    "rolling_velocity_mean",
    "density_rate_of_change",
    "velocity_rate_of_change",
    "density_velocity_ratio",
]

# ── Signage messages (same as predictor.py) ──
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

# ── Model cache (persists across warm Lambda invocations) ──
_model = None
_scaler = None


def _load_model_from_s3():
    """Load model artifacts from S3 at cold start."""
    global _model, _scaler

    if _model is not None and _scaler is not None:
        return

    import boto3
    s3 = boto3.client("s3")

    logger.info(f"Loading model from s3://{S3_BUCKET}/{MODEL_KEY}")
    model_obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)
    _model = pickle.loads(model_obj["Body"].read())

    logger.info(f"Loading scaler from s3://{S3_BUCKET}/{SCALER_KEY}")
    scaler_obj = s3.get_object(Bucket=S3_BUCKET, Key=SCALER_KEY)
    _scaler = pickle.loads(scaler_obj["Body"].read())

    logger.info("✅ Model and scaler loaded successfully")


def _get_risk_level(prob: float) -> tuple:
    """Map probability to risk level and color."""
    if prob >= RED_THRESHOLD:
        return "red", "#FF4444"
    elif prob >= YELLOW_THRESHOLD:
        return "yellow", "#FFAA00"
    else:
        return "green", "#44BB44"


def _get_signage_message(zone_id: str, prob: float) -> tuple:
    """Get signage message based on risk probability."""
    zone_messages = SIGNAGE_MESSAGES.get(zone_id, SIGNAGE_MESSAGES["Zone_A"])

    if prob >= EMERGENCY_THRESHOLD:
        return zone_messages["emergency"], True
    elif prob >= RED_THRESHOLD:
        return zone_messages["critical"], True
    elif prob >= YELLOW_THRESHOLD:
        return zone_messages["warning"], True
    else:
        return "✅ Normal flow. No action required.", False


def _estimate_time_to_congestion(prob: float, density_rate: float, current_density: float) -> float:
    """Estimate minutes until congestion."""
    if prob < YELLOW_THRESHOLD:
        return -1.0

    density_threshold = 4.0
    if density_rate <= 0.01:
        return 2.0 if prob >= RED_THRESHOLD else -1.0

    density_gap = max(density_threshold - current_density, 0.1)
    steps = density_gap / max(density_rate, 0.01)
    minutes = (steps * 30) / 60.0
    minutes = max(0.5, min(minutes, 30.0))

    if prob >= EMERGENCY_THRESHOLD:
        minutes = min(minutes, 2.0)
    elif prob >= RED_THRESHOLD:
        minutes = min(minutes, 8.0)

    return round(minutes, 1)


def _make_response(status_code: int, body: dict) -> dict:
    """Create API Gateway-compatible response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
        "body": json.dumps(body, default=str),
    }


def handler(event, context):
    """
    AWS Lambda handler for congestion prediction.

    Supports:
      POST /predict  — Single zone prediction
      OPTIONS        — CORS preflight
    """
    # Handle CORS preflight
    http_method = event.get("httpMethod", event.get("requestContext", {}).get("http", {}).get("method", "POST"))
    if http_method == "OPTIONS":
        return _make_response(200, {"message": "OK"})

    try:
        # Load model (cached after cold start)
        _load_model_from_s3()

        # Parse request body
        body = event.get("body", "{}")
        if isinstance(body, str):
            body = json.loads(body)

        zone_id = body.get("zone_id", "Zone_A")
        features = body.get("features", {})

        if not features:
            return _make_response(400, {"error": "Missing 'features' in request body"})

        # Build feature vector in correct order
        feature_vector = np.array([[features.get(col, 0) for col in FEATURE_COLUMNS]])

        # Scale and predict
        feature_scaled = _scaler.transform(feature_vector)
        prob = float(_model.predict_proba(feature_scaled)[0][1])

        # Derive outputs
        risk_level, risk_color = _get_risk_level(prob)
        time_to_cong = _estimate_time_to_congestion(
            prob=prob,
            density_rate=features.get("density_rate_of_change", 0),
            current_density=features.get("density", 0),
        )
        message, signage_active = _get_signage_message(zone_id, prob)

        result = {
            "zone_id": zone_id,
            "risk_probability": round(prob, 4),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "time_to_congestion": time_to_cong,
            "signage_message": message,
            "signage_active": signage_active,
        }

        logger.info(f"Prediction for {zone_id}: {prob:.2%} ({risk_level})")
        return _make_response(200, result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return _make_response(500, {"error": f"Prediction failed: {str(e)}"})


# ── Local testing ──
if __name__ == "__main__":
    # Test with a mock event (uses local model files instead of S3)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    # Override _load_model_from_s3 to load locally
    import joblib
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    _model = joblib.load(os.path.join(model_dir, "congestion_model.pkl"))
    _scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))

    test_event = {
        "httpMethod": "POST",
        "body": json.dumps({
            "zone_id": "Zone_A",
            "features": {
                "density": 3.8,
                "velocity": 0.6,
                "rolling_density_mean": 3.2,
                "rolling_velocity_mean": 0.8,
                "density_rate_of_change": 0.35,
                "velocity_rate_of_change": -0.15,
                "density_velocity_ratio": 6.3,
            },
        }),
    }

    result = handler(test_event, None)
    print(json.dumps(json.loads(result["body"]), indent=2))
