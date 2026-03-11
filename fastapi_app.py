"""
# pyre-ignore-all-errors[21]
FastAPI Backend for CrowdAI
===========================
Serves the Machine Learning predictions and the static HTML/JS frontend.
Replaces the Streamlit monolith for better cloud compatibility.
"""

import os
import time
from typing import Dict, Any
import os
import time
import math
import base64
from io import BytesIO
from typing import Dict, Any
from fastapi import FastAPI, BackgroundTasks
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

def _heat_rgba(v):
    # Smoother ROYGBIV to match reference UI exactly
    if v < 0.05: return (0,0,0,0)
    t = max(0., min(1., v))
    
    # HSL-like calculation for smoother gradients.
    # We want: 0.0 -> Blue, 0.4 -> Green, 0.7 -> Yellow, 1.0 -> Red
    if t < 0.35: # Blue to Cyan to Green
        s = t / 0.35
        return (0, int(255*s), int(255*(1-s*0.5)), int(200*s))
    elif t < 0.7: # Green to Yellow
        s = (t - 0.35) / 0.35
        return (int(255*s), 255, 0, min(220, 200 + int(55*s)))
    else: # Yellow to Red
        s = (t - 0.7) / 0.3
        return (255, int(255*(1-s)), 0, 255)

def _build_heat_png(zd,VW=900,VH=520,CR=180,RR=104):
    buf=np.zeros(CR*RR,dtype=np.float32)
    for zid,spots in _HOTSPOTS.items():
        d=float(zd.get(zid,1.5)); w=float(np.clip(d/8.,.05,1.3))
        for cx,cy,sig,base in spots:
            cx_n=cx/VW*CR; cy_n=cy/VH*RR
            sig_n=(sig/VW*CR)*(0.8+w*0.7); amp=base*w
            for r in range(RR):
                for c in range(CR):
                    dx,dy=c-cx_n,r-cy_n
                    buf[r*CR+c]+=amp*math.exp(-(dx*dx+dy*dy)/(2*sig_n*sig_n))
    mx=buf.max()
    if mx>0: buf/=mx
    img=np.zeros((RR,CR,4),dtype=np.uint8)
    for i in range(RR*CR):
        ri,ci=divmod(i,CR)
        rv,gv,bv,av=_heat_rgba(float(buf[i]))
        img[ri,ci]=[rv,gv,bv,int(av)]
    pil=PI.fromarray(img,mode="RGBA").filter(ImageFilter.GaussianBlur(radius=6)) # type: ignore
    bio=BytesIO(); pil.save(bio,format="PNG") # type: ignore
    return "data:image/png;base64,"+base64.b64encode(bio.getvalue()).decode()

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
        global_status: Dict[str, int] = {"low": 0, "warn": 0, "crit": 0}
        result: Dict[str, Any] = {
            "scenario": self.scenario_name,
            "step": self.step,
            "total_steps": self.n_points,
            "time_label": f"{self.step // 60:02d}:{self.step % 60:02d}",
            "zones": {},
            "global_status": global_status
        }
        
        # Temp memory for heatmap generating dict (ZoneID -> density)
        zd_dict = {}
        
        for zone in self.zones:
            hist = self.zone_data[zone].iloc[max(0, self.step - 50):self.step + 1].copy()
            if len(hist) == 0:
                continue
                
            current_row = hist.iloc[-1]
            feats = get_realtime_features(hist)
            
            zone_info: Dict[str, Any] = {
                "density": float(current_row["density"]),
                "velocity": float(current_row["velocity"]),
                "history": {
                    "density": hist["density"].tolist()[-20:],
                    "velocity": hist["velocity"].tolist()[-20:],
                    "labels": [f"{i//60:02d}:{i%60:02d}" for i in range(max(0, self.step-20), self.step)]
                }
            }
            
            if feats:
                pred = predict_zone(zone, feats, self.model, self.scaler)
                zone_info["risk_level"] = pred.risk_level
                zone_info["risk_probability"] = pred.risk_probability * 100
                zone_info["time_to_congestion"] = pred.time_to_congestion
                zone_info["message"] = pred.signage_message
            else:
                zone_info["risk_level"] = "green"
                zone_info["risk_probability"] = 0.0
                zone_info["time_to_congestion"] = 0.0
                zone_info["message"] = "Data collecting..."
                
            # Update global counts
            if zone_info["risk_level"] == "red":
                global_status["crit"] += 1
            elif zone_info["risk_level"] == "yellow":
                global_status["warn"] += 1
            else:
                global_status["low"] += 1
                
            zd_dict[zone] = zone_info["density"]
            result["zones"][zone] = zone_info
            
        result["heatmap_base64"] = _build_heat_png(zd_dict)
            
        return result

# Initialize state
state = SimulationState()

# ── API Endpoints ──

class ScenarioRequest(BaseModel):
    scenario: str

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
    
    for zone in state.zones:
        hist = state.zone_data[zone].iloc[max(0, state.step - 50):state.step + 1].copy()
        if len(hist) == 0:
            continue
            
        current_row = hist.iloc[-1]
        feats = get_realtime_features(hist)
        
        if feats:
            pred = predict_zone(zone, feats, state.model, state.scaler)
            zone_data[zone] = {
                "risk_probability": pred.risk_probability,
                "risk_level": pred.risk_level,
                "density": float(current_row["density"]),
                "velocity": float(current_row["velocity"]),
                "time_to_congestion": pred.time_to_congestion
            }
        else:
            zone_data[zone] = {
                "risk_probability": 0.0,
                "risk_level": "green",
                "density": float(current_row["density"]),
                "velocity": float(current_row["velocity"]),
                "time_to_congestion": 0.0
            }
            
    # Try to generate the overview
    overview = generate_situation_overview(zone_data)
    if not overview:
        overview = """⚠️ **AI Overview Unavailable**
        
Amazon Bedrock could not be reached. Please ensure:
1. Your AWS Credentials are valid and loaded.
2. The IAM Role has `bedrock:InvokeModel`.
3. Model access for **Claude 3 Haiku** is requested and granted in the `us-east-1` region."""
        
    return {"overview": overview}

# ── Serve Static Frontend ──
# Mount the static directory for CSS/JS
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    """Serve the main HTML file."""
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
