"""
AWS Storage Module
==================
Provides integration with:
  1. Amazon S3 — Store and load trained model artifacts (.pkl files)
  2. Amazon DynamoDB — Persist prediction history and sensor readings

Both services gracefully fall back to local storage when AWS is unavailable,
making the system work in both cloud and local development modes.
"""

import os
import io
import json
import time
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Configuration ──
S3_BUCKET = os.environ.get("CROWDAI_S3_BUCKET", "crowdai-models")
DYNAMODB_TABLE = os.environ.get("CROWDAI_DYNAMO_TABLE", "crowdai-predictions")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# ── Lazy-loaded clients ──
_s3_client = None
_dynamodb_resource = None
_s3_available = None
_dynamodb_available = None


# ═══════════════════════════════════════
#  Amazon S3 — Model Artifact Storage
# ═══════════════════════════════════════

def _get_s3_client():
    """Lazy-load S3 client."""
    global _s3_client, _s3_available
    if _s3_client is None:
        try:
            import boto3
            _s3_client = boto3.client("s3", region_name=AWS_REGION)
            # Quick check: list the bucket (will fail if no access)
            _s3_client.head_bucket(Bucket=S3_BUCKET)
            _s3_available = True
            logger.info(f"✅ Amazon S3 connected (bucket: {S3_BUCKET})")
        except Exception as e:
            _s3_available = False
            logger.warning(f"⚠️ Amazon S3 unavailable: {e}")
    return _s3_client


def is_s3_available() -> bool:
    """Check if S3 is available."""
    global _s3_available
    if _s3_available is None:
        _get_s3_client()
    return _s3_available or False


def upload_model_to_s3(local_path: str, s3_key: str) -> bool:
    """
    Upload a model artifact to S3.
    Returns True if successful, False otherwise.
    """
    client = _get_s3_client()
    if client is None:
        return False

    try:
        client.upload_file(local_path, S3_BUCKET, s3_key)
        logger.info(f"✅ Uploaded {local_path} → s3://{S3_BUCKET}/{s3_key}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ S3 upload failed: {e}")
        return False


def download_model_from_s3(s3_key: str, local_path: str) -> bool:
    """
    Download a model artifact from S3 to local path.
    Returns True if successful, False otherwise.
    """
    client = _get_s3_client()
    if client is None:
        return False

    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        client.download_file(S3_BUCKET, s3_key, local_path)
        logger.info(f"✅ Downloaded s3://{S3_BUCKET}/{s3_key} → {local_path}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ S3 download failed: {e}")
        return False


def load_model_bytes_from_s3(s3_key: str) -> Optional[bytes]:
    """
    Load model bytes directly from S3 (for Lambda cold-start).
    Returns bytes if successful, None otherwise.
    """
    client = _get_s3_client()
    if client is None:
        return None

    try:
        response = client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        return response["Body"].read()
    except Exception as e:
        logger.warning(f"⚠️ S3 read failed for {s3_key}: {e}")
        return None


# ═══════════════════════════════════════
#  Amazon DynamoDB — Prediction History
# ═══════════════════════════════════════

def _get_dynamodb_table():
    """Lazy-load DynamoDB table resource."""
    global _dynamodb_resource, _dynamodb_available
    if _dynamodb_resource is None:
        try:
            import boto3
            dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
            _dynamodb_resource = dynamodb.Table(DYNAMODB_TABLE)
            # Quick check: describe the table
            _dynamodb_resource.table_status
            _dynamodb_available = True
            logger.info(f"✅ DynamoDB connected (table: {DYNAMODB_TABLE})")
        except Exception as e:
            _dynamodb_available = False
            logger.warning(f"⚠️ DynamoDB unavailable: {e}")
    return _dynamodb_resource


def is_dynamodb_available() -> bool:
    """Check if DynamoDB is available."""
    global _dynamodb_available
    if _dynamodb_available is None:
        _get_dynamodb_table()
    return _dynamodb_available or False


def store_prediction(
    zone_id: str,
    timestamp: str,
    risk_probability: float,
    risk_level: str,
    density: float,
    velocity: float,
    time_to_congestion: float,
    signage_message: str,
    scenario: str = "unknown",
) -> bool:
    """
    Store a prediction record in DynamoDB.

    Table schema:
      PK: zone_id (String)
      SK: timestamp (String, ISO format)
      Attributes: risk_probability, risk_level, density, velocity,
                  time_to_congestion, signage_message, scenario

    Returns True if successful, False otherwise.
    """
    table = _get_dynamodb_table()
    if table is None:
        return False

    try:
        from decimal import Decimal

        table.put_item(Item={
            "zone_id": zone_id,
            "timestamp": timestamp,
            "risk_probability": Decimal(str(round(risk_probability, 4))),
            "risk_level": risk_level,
            "density": Decimal(str(round(density, 2))),
            "velocity": Decimal(str(round(velocity, 2))),
            "time_to_congestion": Decimal(str(round(time_to_congestion, 1))),
            "signage_message": signage_message,
            "scenario": scenario,
            "ttl": int(time.time()) + 86400,  # Auto-expire after 24 hours
        })
        return True
    except Exception as e:
        logger.warning(f"⚠️ DynamoDB write failed: {e}")
        return False


def get_prediction_history(
    zone_id: str,
    limit: int = 50,
) -> list[dict]:
    """
    Retrieve recent prediction history for a zone from DynamoDB.
    Returns list of prediction records, newest first.
    """
    table = _get_dynamodb_table()
    if table is None:
        return []

    try:
        from boto3.dynamodb.conditions import Key

        response = table.query(
            KeyConditionExpression=Key("zone_id").eq(zone_id),
            ScanIndexForward=False,  # Newest first
            Limit=limit,
        )
        return response.get("Items", [])
    except Exception as e:
        logger.warning(f"⚠️ DynamoDB query failed: {e}")
        return []


def store_incident(
    incident_id: str,
    zone_data: dict,
    summary: str,
    scenario: str = "unknown",
) -> bool:
    """
    Store an incident record when Red alert is triggered.
    Uses a separate sort key prefix for incidents.
    """
    table = _get_dynamodb_table()
    if table is None:
        return False

    try:
        from decimal import Decimal

        table.put_item(Item={
            "zone_id": "INCIDENT",
            "timestamp": incident_id,
            "zones_affected": json.dumps(zone_data, default=str),
            "summary": summary,
            "scenario": scenario,
            "created_at": datetime.utcnow().isoformat(),
            "ttl": int(time.time()) + 604800,  # 7 days
        })
        return True
    except Exception as e:
        logger.warning(f"⚠️ DynamoDB incident write failed: {e}")
        return False


# ═══════════════════════════════════════
#  AWS Status Summary
# ═══════════════════════════════════════

def get_aws_status() -> dict:
    """
    Return status of all AWS service connections.
    Used by the dashboard to show AWS integration status.
    """
    from src.aws_bedrock import is_bedrock_available

    return {
        "s3": is_s3_available(),
        "dynamodb": is_dynamodb_available(),
        "bedrock": is_bedrock_available(),
    }
