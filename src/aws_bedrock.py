"""
AWS Bedrock Integration Module
==============================
Uses Amazon Bedrock (Claude 3 Haiku) to generate:
  1. Dynamic digital signage messages — context-aware crowd guidance
  2. Incident summaries — natural-language briefs for event organizers

Why AI is required:
  Static templates can't adapt to nuanced situations (e.g., cascading
  congestion across multiple zones, varying severity levels). An LLM
  generates situation-specific guidance that's more actionable.

What value the AI layer adds:
  - Signage messages adapt to real-time conditions (density, velocity, zone)
  - Incident briefs synthesize multi-zone data into actionable summaries
  - Emergency messaging is more specific and calmer than hardcoded templates
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Bedrock client (lazy-loaded) ──
_bedrock_client = None
_bedrock_available = None

MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
AWS_REGION = "us-east-1"


def _get_bedrock_client():
    """Lazy-load the Bedrock runtime client."""
    global _bedrock_client, _bedrock_available
    if _bedrock_client is None:
        try:
            import boto3
            from botocore.config import Config

            config = Config(
                connect_timeout=3,
                read_timeout=10,
                retries={"max_attempts": 1},
            )
            client = boto3.client(
                "bedrock-runtime",
                region_name=AWS_REGION,
                config=config,
            )
            # Verify credentials are configured
            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials is None:
                raise Exception("No AWS credentials configured")
            _bedrock_client = client
            _bedrock_available = True
            logger.info("✅ Amazon Bedrock client initialized")
        except Exception as e:
            _bedrock_available = False
            logger.warning(f"⚠️ Amazon Bedrock unavailable: {e}")
    return _bedrock_client
    return _bedrock_client


def is_bedrock_available() -> bool:
    """Check if Bedrock is available and configured."""
    global _bedrock_available
    if _bedrock_available is None:
        _get_bedrock_client()
    return _bedrock_available or False


def _invoke_bedrock(prompt: str, max_tokens: int = 100) -> Optional[str]:
    """
    Send a prompt to Amazon Bedrock and return the response text.
    Returns None if Bedrock is unavailable or the call fails.
    """
    client = _get_bedrock_client()
    if client is None:
        return None

    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": 0.3,  # Low temp for consistent, safe messaging
            "messages": [{"role": "user", "content": prompt}],
        })

        response = client.invoke_model(
            modelId=MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())
        return result["content"][0]["text"].strip()

    except Exception as e:
        logger.warning(f"⚠️ Bedrock invocation failed: {e}")
        return None


def generate_signage_message(
    zone_id: str,
    risk_level: str,
    risk_probability: float,
    density: float,
    velocity: float,
    time_to_congestion: float,
) -> Optional[str]:
    """
    Use Amazon Bedrock to generate a context-aware digital signage message
    based on real-time crowd conditions.

    Returns None if Bedrock is unavailable (caller should use fallback).
    """
    prompt = f"""You are a crowd safety system generating a SHORT digital signage message 
(max 20 words) for people in {zone_id} of a venue.

Current conditions:
- Risk level: {risk_level}
- Congestion probability: {risk_probability:.0%}
- Crowd density: {density:.1f} people/m²
- Movement speed: {velocity:.2f} m/s
- Estimated time to congestion: {time_to_congestion:.0f} minutes

Generate ONE clear, calm, actionable message for the digital sign.
No panic language. Be specific about directions when possible.
Do not include any prefix, label, or emoji — just the message text."""

    return _invoke_bedrock(prompt, max_tokens=60)


def generate_incident_summary(zone_data: dict) -> Optional[str]:
    """
    Generate a natural-language incident brief for event organizers
    using Bedrock when a Red alert is triggered.

    Args:
        zone_data: dict mapping zone_id to {risk_probability, risk_level,
                   density, velocity, time_to_congestion}

    Returns None if Bedrock is unavailable.
    """
    # Format zone data for the prompt
    zone_summary_lines = []
    for zone_id, data in zone_data.items():
        zone_summary_lines.append(
            f"  - {zone_id}: risk={data['risk_probability']:.0%} "
            f"({data['risk_level']}), density={data['density']:.1f} p/m², "
            f"velocity={data['velocity']:.2f} m/s, "
            f"time_to_congestion={data['time_to_congestion']:.0f} min"
        )
    zone_summary = "\n".join(zone_summary_lines)

    prompt = f"""You are a crowd safety AI analyst. Write a concise 3-sentence incident brief 
for event organizers based on this real-time data:

{zone_summary}

Include: (1) what's happening, (2) which zones are affected and severity, 
(3) recommended immediate action. Be factual and actionable. No markdown formatting."""

    return _invoke_bedrock(prompt, max_tokens=150)


def generate_crowd_recommendation(
    zone_data: dict,
    scenario_name: str,
) -> Optional[str]:
    """
    Generate an overall crowd management recommendation using Bedrock.
    Used in the dashboard sidebar for organizer guidance.
    """
    zone_summary_lines = []
    for zone_id, data in zone_data.items():
        zone_summary_lines.append(
            f"  - {zone_id}: risk={data['risk_probability']:.0%}, "
            f"density={data['density']:.1f}, velocity={data['velocity']:.2f}"
        )
    zone_summary = "\n".join(zone_summary_lines)

    prompt = f"""You are a crowd management AI advisor. Given this scenario ({scenario_name}) 
and current zone conditions:

{zone_summary}

Provide ONE short recommendation (max 2 sentences) for the event organizer. 
Be specific about which zones need attention and what action to take."""

    return _invoke_bedrock(prompt, max_tokens=80)
