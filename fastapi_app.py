"""
# pyre-ignore-all-errors[21]
FastAPI Backend for CrowdAI
===========================
Serves the Machine Learning predictions and the static HTML/JS frontend.
Replaces the Streamlit monolith for better cloud compatibility.
"""

import os
import time
import math
import base64
import asyncio
from io import BytesIO
from typing import Dict, Any, List, Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from PIL import Image as PI, ImageFilter

# Import our existing ML logic
from src.simulate_data import (
    generate_normal_day,
    generate_post_event_rush,
    generate_emergency_evacuation,
)
from src.features import get_realtime_features, ROLLING_WINDOW
from src.predictor import predict_zone
from src.model import load_model

# ── Custom Zone Config (in-memory, cleared on restart) ──
# Shape: { "map_image": "data:image/...base64...", "zones": [{name,x,y,w,h}, ...] }
CUSTOM_CONFIG: Optional[Dict] = None

def _normalize_name(name: str) -> str:
    """Normalize names for robust matching: lowercase, no spaces or underscores."""
    return "".join(name.split()).replace("_", "").lower()

app = FastAPI(title="CrowdAI Backend")

# ── Globals for Heatmap (Rescaled to 900x520 bounds for better bleed) ──
_HOTSPOTS = {
    # Zone A (Y: 0-140)
    "Zone_A":[(140,80,65,.9),(330,65,55,.6),(580,90,50,.5),(750,80,45,.4)],
    # Zone B (Y: 140-280)
    "Zone_B":[(200,200,75,.85),(450,190,65,.75),(700,220,60,.6)],
    # Zone C (Y: 280-520)
    "Zone_C":[(200,400,65,.8),(450,440,55,.6),(700,380,50,.55)],
}
_MAX_DENSITY = 10.0 # Match simulation max

def _heat_rgba(v):
    # Standard High-Contrast Thermal Scale
    if v < 0.05: return (0, 0, 0, 0) # Clear transparency for empty space
    t = max(0., min(1., v))
    
    # Adaptive alpha: semi-transparent for low heat, solid for high heat
    alpha = int(100 + 155 * t)
    
    if t < 0.25: # Cool (Deep Blue to Cyan)
        s = t / 0.25
        return (0, int(150 * s), 255, alpha)
    elif t < 0.5: # Moderate (Cyan to Green)
        s = (t - 0.25) / 0.25
        return (0, 255, int(255 * (1 - s)), alpha)
    elif t < 0.75: # Warning (Green to Yellow)
        s = (t - 0.5) / 0.25
        return (int(255 * s), 255, 0, alpha)
    else: # Danger (Yellow to Deep Red)
        s = (t - 0.75) / 0.25
        return (255, int(255 * (1 - s)), 0, alpha)

def _build_heat_png(zd, VW=900, VH=520, CR=180, RR=104):
    """Default heatmap using predefined hotspots for Zone_A/B/C."""
    # Create a normalized lookup map for current data
    zd_norm = { _normalize_name(k): v for k, v in zd.items() }
    
    buf = np.zeros(CR * RR, dtype=np.float32)
    for zid, spots in _HOTSPOTS.items():
        # Match using normalized ID
        d = float(zd_norm.get(_normalize_name(zid), 0.0))
        # Use a slight non-linear boost for visibility of lower densities
        w = float(np.clip(math.pow(d / _MAX_DENSITY, 0.7), 0.0, 1.0))
        for cx, cy, sig, base in spots:
            cx_n = cx / VW * CR
            cy_n = cy / VH * RR
            sig_n = (sig / VW * CR) * (0.8 + w * 0.4)
            amp = base * w * 2.5 # Increased base amplitude
            # Vectorized Gaussian
            cols = np.arange(CR, dtype=np.float32)
            rows = np.arange(RR, dtype=np.float32)
            dc = cols - cx_n
            dr = rows - cy_n
            buf += amp * np.exp(-(dc[np.newaxis, :] ** 2 + dr[:, np.newaxis] ** 2) / (2 * sig_n ** 2)).ravel()
    
    buf = np.clip(buf, 0.0, 1.0)
    img = np.zeros((RR, CR, 4), dtype=np.uint8)
    for i in range(RR * CR):
        ri, ci = divmod(i, CR)
        img[ri, ci] = list(_heat_rgba(float(buf[i])))
    # Reduced blur radius to keep spots distinct
    pil = PI.fromarray(img, mode="RGBA").filter(ImageFilter.GaussianBlur(radius=3))  # type: ignore
    bio = BytesIO()
    pil.save(bio, format="PNG")  # type: ignore
    return "data:image/png;base64," + base64.b64encode(bio.getvalue()).decode()


def _build_custom_heat(custom_zones: List[Dict], zd: Dict[str, float], VW=900, VH=520, CR=180, RR=104):
    """Heatmap for user-drawn zones: Fills the rectangular area of each zone, scaled by density."""
    # Create a normalized lookup map
    zd_norm = { _normalize_name(k): v for k, v in zd.items() }

    buf = np.zeros((RR, CR), dtype=np.float32)
    for z in custom_zones:
        # Scale coordinates from 900x520 to CRxRR
        x1 = int(max(0, z['x'] / VW * CR))
        y1 = int(max(0, z['y'] / VH * RR))
        x2 = int(min(CR, (z['x'] + z['w']) / VW * CR))
        y2 = int(min(RR, (z['y'] + z['h']) / VH * RR))
        
        # Robust lookup
        zname_norm = _normalize_name(z['name'])
        density = float(zd_norm.get(zname_norm, 0.0))
        # Boost visibility for non-empty areas
        val = float(np.clip(math.pow(density / _MAX_DENSITY, 0.6), 0.0, 1.2))
        
        # Fill the zone area - using max to handle overlapping zones cleanly
        if x2 > x1 and y2 > y1:
            buf[y1:y2, x1:x2] = np.maximum(buf[y1:y2, x1:x2], val)

    # Convert to RGBA
    img = np.zeros((RR, CR, 4), dtype=np.uint8)
    for r in range(RR):
        for c in range(CR):
            img[r, c] = list(_heat_rgba(float(buf[r, c])))
            
    # Apply a moderate blur (8 was too much and washed out low heat)
    pil = PI.fromarray(img, mode="RGBA").filter(ImageFilter.GaussianBlur(radius=4))  # type: ignore
    bio = BytesIO()
    pil.save(bio, format="PNG")  # type: ignore
    return "data:image/png;base64," + base64.b64encode(bio.getvalue()).decode()

# ── Global State (Simulation) ──
class SimulationState:
    def __init__(self):
        self.scenario_name = "🏢 Normal Day"
        self.step = ROLLING_WINDOW
        self.model, self.scaler = load_model()
        self.full_data: pd.DataFrame = pd.DataFrame()
        self.zones: list[str] = []
        self.zone_data: dict[str, pd.DataFrame] = {}
        self.n_points: int = 0
        
        # Scenario map
        self.scenarios = {
            "🏢 Normal Day": generate_normal_day,
            "🎉 Post-Event Rush": generate_post_event_rush,
            "🚨 Emergency Evacuation": generate_emergency_evacuation,
        }
        self.load_scenario(self.scenario_name)

    def load_scenario(self, name: str):
        np.random.seed(42)
        self.scenario_name = name
        self.full_data = self.scenarios[name]()
        self.zones = sorted(self.full_data["zone_id"].unique())
        self.zone_data = {
            z: self.full_data[self.full_data["zone_id"] == z].reset_index(drop=True)
            for z in self.zones
        }
        self.step = ROLLING_WINDOW
        self.n_points = len(self.zone_data[self.zones[0]])

    def tick(self):
        self.step += 1
        if self.step >= self.n_points:
            self.step = ROLLING_WINDOW

    def get_current_data(self) -> Dict[str, Any]:
        global CUSTOM_CONFIG

        # ── Determine active zones ──
        is_custom = False
        custom_zones: List[Dict] = []
        if CUSTOM_CONFIG and isinstance(CUSTOM_CONFIG.get('zones'), list):
            valid = [z for z in CUSTOM_CONFIG['zones']
                     if z.get('name') and z.get('w', 0) > 5 and z.get('h', 0) > 5]
            if valid:
                custom_zones = valid
                is_custom = True

        # Source zone names for ML data (cycle through Zone_A/B/C)
        base_zones = self.zones  # ['Zone_A', 'Zone_B', 'Zone_C']
        active_zones = [z['name'] for z in custom_zones] if is_custom else base_zones

        global_status: Dict[str, int] = {"low": 0, "warn": 0, "crit": 0}
        result: Dict[str, Any] = {
            "scenario": self.scenario_name,
            "step": self.step,
            "total_steps": self.n_points,
            "time_label": f"{self.step // 60:02d}:{self.step % 60:02d}",
            "zones": {},
            "global_status": global_status,
            "is_custom": is_custom,
            "config": CUSTOM_CONFIG if is_custom else None,
        }

        zd_dict: Dict[str, float] = {}

        for i, zone_name in enumerate(active_zones):
            # Map custom zone to a real simulation zone (cycle)
            src_zone = base_zones[i % len(base_zones)]
            hist = self.zone_data[src_zone].iloc[max(0, self.step - 50):self.step + 1].copy()
            if len(hist) == 0:
                continue

            current_row = hist.iloc[-1]
            feats = get_realtime_features(hist)

            density = float(current_row["density"])
            # Remove jitter to ensure 1:1 match between Dashboard numbers and Heatmap intensity

            zone_info: Dict[str, Any] = {
                "density": density,
                "velocity": float(current_row["velocity"]),
                "history": {
                    "density": hist["density"].tolist()[-20:],
                    "velocity": hist["velocity"].tolist()[-20:],
                    "labels": [f"{j//60:02d}:{j%60:02d}" for j in range(max(0, self.step-20), self.step)]
                }
            }

            if feats:
                pred = predict_zone(src_zone, feats, self.model, self.scaler)
                zone_info["risk_level"] = pred.risk_level
                zone_info["risk_probability"] = pred.risk_probability * 100
                zone_info["time_to_congestion"] = pred.time_to_congestion
                zone_info["message"] = pred.signage_message
            else:
                zone_info["risk_level"] = "green"
                zone_info["risk_probability"] = 0.0
                zone_info["time_to_congestion"] = 0.0
                zone_info["message"] = "Data collecting..."

            if zone_info["risk_level"] == "red":
                global_status["crit"] += 1
            elif zone_info["risk_level"] == "yellow":
                global_status["warn"] += 1
            else:
                global_status["low"] += 1

            zd_dict[_normalize_name(zone_name)] = zone_info["density"]
            result["zones"][zone_name] = zone_info

        # Generate heatmap
        if is_custom:
            result["heatmap_base64"] = _build_custom_heat(custom_zones, zd_dict)
        else:
            result["heatmap_base64"] = _build_heat_png(zd_dict)

        return result

# Initialize state
state = SimulationState()

# ── API Endpoints ──

class ScenarioRequest(BaseModel):
    scenario: str

class ZoneConfigRequest(BaseModel):
    map_image: Optional[str] = None
    zones: List[Dict]

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/data")
def get_dashboard_data():
    """Returns the current step data for all zones."""
    return state.get_current_data()

@app.post("/api/tick")
def advance_simulation():
    """Advances the simulation by 1 step (called by frontend interval)."""
    state.tick()
    return {"status": "advanced", "step": state.step}

@app.post("/api/config/zones")
def save_zone_config(req: ZoneConfigRequest):
    """Store a custom map image and zone definitions."""
    global CUSTOM_CONFIG
    valid = [z for z in req.zones if z.get('name') and z.get('w', 0) > 5 and z.get('h', 0) > 5]
    if not valid:
        return {"status": "error", "message": "No valid zones provided"}
    CUSTOM_CONFIG = {"map_image": req.map_image, "zones": valid}
    print(f"[Config] Custom layout saved: {len(valid)} zone(s)")
    return {"status": "success", "zones_saved": len(valid)}

@app.post("/api/config/reset")
def reset_zone_config():
    """Clear custom configuration, revert to default zones."""
    global CUSTOM_CONFIG
    CUSTOM_CONFIG = None
    print("[Config] Custom layout cleared.")
    return {"status": "cleared"}

@app.post("/api/scenario")
def set_scenario(req: ScenarioRequest):
    """Changes the active scenario."""
    if req.scenario in state.scenarios:
        state.load_scenario(req.scenario)
        return {"status": "success", "scenario": req.scenario}
    return {"status": "error", "message": "Unknown scenario"}

@app.post("/api/reset")
def reset_simulation():
    """Resets the current scenario to the beginning."""
    state.load_scenario(state.scenario_name)
    return {"status": "reset", "step": state.step}

@app.get("/api/overview")
def get_ai_overview():
    """Fetches a detailed AI overview from Amazon Bedrock based on current state."""
    zone_data = {}
    from src.aws_bedrock import generate_situation_overview
    
    # Get the same data as the dashboard to ensure consistency
    current_data = state.get_current_data()
    active_zones = current_data.get("zones", {})
    
    for zone_id, data in active_zones.items():
        # Map back to simulation zone for history
        src_map = { _normalize_name(z): z for z in state.zones }
        src_zone = src_map.get(_normalize_name(zone_id), state.zones[0])
        
        hist = state.zone_data[src_zone].iloc[max(0, state.step - 50):state.step + 1].copy()
        if len(hist) == 0:
            continue
            
        current_row = hist.iloc[-1]
        feats = get_realtime_features(hist)
        
        if feats:
            zone_data[zone_id] = {
                "risk_probability": data.get("risk_probability", 0),
                "risk_level": data.get("risk_level", "green"),
                "density": data.get("density", 0.0),
                "velocity": data.get("velocity", 0.0),
                "time_to_congestion": data.get("time_to_congestion", 0)
            }
        else:
            zone_data[zone_id] = {
                "risk_probability": 0.0,
                "risk_level": "green",
                "density": data.get("density", 0.0),
                "velocity": data.get("velocity", 0.0),
                "time_to_congestion": 0.0
            }
            
    # Try to generate the overview
    try:
        overview = generate_situation_overview(zone_data)
        if not overview:
            # This case usually means internal error or empty result
            overview = "⚠️ AI Overview received an empty response from Bedrock."
    except Exception as e:
        err_msg = str(e)
        overview = f"""⚠️ **AI Overview Unavailable**
        
**Technical Error:** {err_msg}

**Troubleshooting Steps:**
1. **IAM Role:** Does your App Runner Instance Role have `bedrock:InvokeModel`?
2. **Model Access:** Go to AWS Console -> Bedrock -> Model Catalog. Ensure 'Amazon Nova Lite' is enabled.
3. **Region:** Ensure you are using `us-east-1` (N. Virginia)."""
        
    return {"overview": overview}

# ── Serve Static Frontend ──
# Mount the static directory for CSS/JS
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    """Serve the main HTML file."""
    return FileResponse("static/index.html")

@app.get("/mobile")
def serve_mobile():
    """Serve the mobile guard app."""
    return FileResponse("static/mobile.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
