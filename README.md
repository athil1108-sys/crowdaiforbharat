# 🏟️ CrowdAI — Privacy-First Congestion Prediction System

An ML-powered crowd management system that predicts congestion **10-15 minutes before it happens** using simulated mmWave radar sensor data. Built as a software-only prototype — no hardware required.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)

---

## 📁 Project Structure

```
crowd-management/
├── app.py                  # Streamlit dashboard (main entry point)
├── run.sh                  # One-click startup script
├── requirements.txt        # Python dependencies
├── README.md
├── src/
│   ├── __init__.py
│   ├── simulate_data.py    # Sensor data simulation (3 scenarios)
│   ├── features.py         # Feature engineering pipeline
│   ├── model.py            # ML model training & evaluation
│   ├── predictor.py        # Real-time prediction logic
│   ├── aws_bedrock.py      # Amazon Bedrock signage & incident briefs
│   ├── aws_storage.py      # S3 model storage & DynamoDB history
│   └── lambda_handler.py   # AWS Lambda prediction endpoint
├── models/
│   ├── congestion_model.pkl  # Trained Logistic Regression model
│   └── scaler.pkl            # Feature scaler
└── venv/                   # Python virtual environment
```

---

## 🚀 How to Run

### Quick Start
```bash
# Make sure you're in the project directory
cd crowd-management

# Option 1: Use the run script
./run.sh

# Option 2: Manual steps
source venv/bin/activate
python -m src.model          # Train the model (first time only)
streamlit run app.py         # Launch the dashboard
```

Then open **http://localhost:8501** in your browser.

### From Scratch
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.model
streamlit run app.py
```

---

## 🧠 How It Works

### 1. Data Simulation (`src/simulate_data.py`)
Simulates what mmWave radar sensors (LD2410 + ESP32) would send:
- **3 zones** (A, B, C) with independent sensor feeds
- Each data point: `zone_id`, `timestamp`, `density`, `velocity`
- Realistic Gaussian noise added to simulate sensor imperfections
- **3 scenarios:**
  - 🏢 **Normal Day** — Mild congestion at lunch/EOD
  - 🎉 **Post-Event Rush** — Cascading congestion across all zones
  - 🚨 **Emergency Evacuation** — Simultaneous spikes everywhere

### 2. Feature Engineering (`src/features.py`)
Transforms raw data into 7 predictive features:

| Feature | Why It Helps |
|---------|-------------|
| `rolling_density_mean` | Smooths noise; sustained buildup signal |
| `rolling_velocity_mean` | Sustained slowdown detection |
| `density_rate_of_change` | How fast crowd is building |
| `velocity_rate_of_change` | How fast people are stopping |
| `density_velocity_ratio` | Combined congestion indicator |
| `density` | Raw current crowd level |
| `velocity` | Raw current movement speed |

### 3. ML Model (`src/model.py`)
- **Algorithm:** Logistic Regression with balanced class weights
- **Training data:** 7,530 samples across 13 simulation runs
- **Prediction target:** "Will congestion happen in next 12.5 minutes?"

**Performance:**
| Metric | Score |
|--------|-------|
| Accuracy | 92.0% |
| Precision | 91.8% |
| Recall | 82.6% |
| F1 Score | 86.9% |

### 4. Prediction Logic (`src/predictor.py`)
For each zone, outputs:
- **Risk Probability** (0-100%)
- **Risk Level:** 🟢 Green (<40%) / 🟡 Yellow (40-70%) / 🔴 Red (>70%)
- **Time to Congestion** (estimated minutes)
- **Digital Signage Message** (auto-triggered when risk > 70%)

### 5. Dashboard (`app.py`)
Streamlit real-time dashboard with:
- Live density & velocity charts per zone
- Risk probability gauges
- Color-coded risk cards
- Auto-updating digital signage
- Scenario switching
- 2-second refresh rate

---

## 🎤 Hackathon Demo Script

### Opening (30 seconds)
> "We built CrowdAI — an AI system that predicts crowd congestion 10-15 minutes before it happens, giving organizers time to act. In real deployment, this uses mmWave radar for complete privacy — no cameras, no facial recognition."

### Demo Flow (2-3 minutes)
1. **Start with Normal Day** — Show green indicators, normal flow
2. **Wait for first congestion** — Watch density rise, velocity drop
3. **Point out the prediction** — "See? The model flagged Yellow BEFORE the spike"
4. **Show the signage trigger** — "At 70% risk, it automatically sends redirect messages"
5. **Switch to Emergency** — "Here's worst case — all zones spike, instant red alerts"

### Closing (30 seconds)
> "The key innovation: we predict congestion, not just detect it. 10-15 minutes of warning means you can redirect crowds BEFORE bottlenecks form. And with mmWave radar, it's completely privacy-preserving — no personal data ever collected."

---

## 🔬 Technical Details

### Congestion Physics
Congestion occurs when:
- Density rises above **4.0 people/m²**
- Velocity drops below **0.5 m/s**
- The combination creates a self-reinforcing bottleneck

### Prediction vs Detection
- **Detection** = "There IS congestion right now" (too late)
- **Prediction** = "There WILL BE congestion in 12.5 minutes" (actionable)

We achieve prediction by shifting the target label backward in time, so the model learns to recognize **precursor patterns** (gradual density increase + velocity decrease) before the actual congestion threshold is crossed.

---

## ☁️ AWS Architecture

### System Architecture
```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌───────────┐
│  Streamlit App   │───▶│ API Gateway  │───▶│  AWS Lambda     │───▶│ DynamoDB  │
│  (AWS Amplify)   │    │  (REST API)  │    │  (Prediction)   │    │ (History) │
└─────────────────┘    └──────────────┘    └────────┬────────┘    └───────────┘
                                                    │
                                           ┌────────▼────────┐    ┌───────────┐
                                           │ Amazon Bedrock   │    │ Amazon S3 │
                                           │ (Claude 3 Haiku) │    │ (Models)  │
                                           └─────────────────┘    └───────────┘
```

### Why AI is Required
Static rule-based systems can only **detect** congestion after it happens. Our ML model **predicts** congestion 10-15 minutes ahead by learning precursor patterns in density/velocity data. Amazon Bedrock adds a second AI layer — generating context-aware signage messages and incident briefs that adapt to the specific situation rather than using rigid templates.

### AWS Services Used

| Service | Purpose | Implementation |
|---------|---------|----------------|
| **Amazon Bedrock** (Claude 3 Haiku) | Generate dynamic digital signage messages & incident summaries | `src/aws_bedrock.py` |
| **AWS Lambda** | Serverless prediction endpoint — scales to thousands of sensors | `src/lambda_handler.py` |
| **Amazon API Gateway** | REST API fronting the prediction Lambda | POST `/predict` endpoint |
| **Amazon S3** | Store trained model artifacts (`.pkl` files) | `src/aws_storage.py` |
| **Amazon DynamoDB** | Persist historical readings & prediction audit trail | `src/aws_storage.py` |
| **AWS Amplify** | Host the Streamlit dashboard | Production deployment |

### What Value the AI Layer Adds
1. **Predictive ML model** — 10-15 min early warning (92% accuracy) vs reactive detection
2. **Bedrock LLM** — Situation-specific crowd guidance instead of generic "area full" messages
3. **Bedrock incident briefs** — Instant natural-language summaries for organizers during emergencies
4. **Graceful fallback** — System works with static templates when Bedrock is unavailable

### AWS Integration Files
```
src/
├── aws_bedrock.py     # Bedrock signage generation & incident briefs
├── aws_storage.py     # S3 model storage & DynamoDB prediction history
└── lambda_handler.py  # Serverless prediction endpoint for API Gateway
```

---

## 📄 License

MIT — Built for hackathon demonstration purposes.
