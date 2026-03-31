# Sentient

**Passive human presence detection using WiFi signal variance, machine learning, and computer vision — no specialized hardware required.**

Sentient monitors the WiFi signals already present in your environment and learns to distinguish between an occupied and an empty room. When a person moves through a space, their body absorbs and reflects radio waves, causing tiny fluctuations in the RSSI (signal strength) of nearby networks. Sentient detects, classifies, and logs these fluctuations in real time.

No cameras required. No ESP32. No cloud. Just your laptop WiFi and a trained model built from your own room's signal data.

---

## How it works

```
WiFi networks in range
        ↓
RSSI sampled every 2 seconds
        ↓
Kalman filter applied per network (noise reduction)
        ↓
12 features extracted per snapshot
(mean, variance, delta, rolling std-dev, band counts...)
        ↓
Random Forest classifier (trained on your room's data)
        ↓
Prediction: occupied / empty + confidence score
        ↓
Activity classification: absent → still → active → walking
        ↓
Live dashboard + SQLite logging + CSV export
```

---

## Features

- **WiFi-based presence detection** — RSSI variance analysis across all visible networks
- **Kalman filtering** — per-network noise reduction for cleaner signal readings
- **2.4GHz + 5GHz tracking** — both bands monitored and tracked independently
- **ML classifier** — Random Forest trained on your own room's signal data
- **Activity classification** — absent / still / active / walking
- **Confidence scoring** — 0–100% confidence on every prediction
- **Zone estimation** — near / mid / far based on signal strength
- **Webcam person count** — OpenCV HOG detector, runs every 10 seconds
- **SQLite persistence** — all events logged with timestamp, activity, confidence
- **Hourly occupancy heatmap** — visual breakdown of presence across the day
- **CSV export** — download any time window as a spreadsheet
- **Desktop notifications** — alerts on presence detected / area cleared
- **Separate frontend and backend** — FastAPI REST API + standalone HTML dashboard

---

## Project structure

```
sentient/
├── backend/
│   ├── main.py              ← FastAPI server, detection engine
│   └── requirements.txt
├── frontend/
│   └── index.html           ← Dashboard (open directly in browser)
├── ml/
│   ├── record.py            ← Phase 1: collect training data
│   ├── train.py             ← Phase 2: train the classifier
│   ├── training_data.csv    ← generated after recording
│   ├── model.pkl            ← generated after training
│   └── model_meta.json      ← accuracy, features, timestamp
└── README.md
```

---

## Requirements

- Python 3.8+
- WiFi adapter (any standard laptop WiFi)
- Windows 10+ / Linux / macOS
- Webcam (optional, for person count)

---

## Installation

```bash
pip install fastapi uvicorn numpy opencv-python-headless scipy scikit-learn pandas joblib
```

For desktop notifications on Windows:
```bash
pip install plyer
```

---

## Quickstart

### 1. Start the backend

```bash
# Windows (run as Administrator for WiFi scanning)
python backend/main.py

# Linux / macOS
sudo python3 backend/main.py
```

Backend runs at `http://localhost:8000`
API docs at `http://localhost:8000/docs`

### 2. Open the dashboard

Open `frontend/index.html` directly in any browser. No build step needed.

---

## Training the ML model

The ML model is optional but significantly improves accuracy. Without it, Sentient falls back to variance-threshold detection.

### Step 1 — Record empty room (3 minutes)

```bash
python ml/record.py empty
```

Leave the room while this runs.

### Step 2 — Record occupied room (3 minutes)

```bash
python ml/record.py occupied
```

Stay in the room. Move around, sit, type — normal activity.

### Step 3 — Train

```bash
python ml/train.py
```

Output example:
```
Cross-validation accuracy: 0.891 ± 0.031
Test accuracy: 0.873 (87.3%)

Feature importances:
  rolling_variance          0.284  ███████████
  mean_delta                0.201  ████████
  rolling_mean_delta        0.187  ███████
  std_signal                0.098  ████
  ...
```

### Step 4 — Restart the backend

```bash
python backend/main.py
```

The backend auto-loads `model.pkl` on startup and prints:
```
[ML] Model loaded — accuracy 87.3%
```

### Improving accuracy

- Record more data — 10+ minutes per class is better than 3
- Record across different sessions (morning, evening, different days)
- Record different activities: sitting still, typing, walking, watching video
- Each `record.py` run appends to `training_data.csv`, then retrain

Typical accuracy ranges:
| Data collected | Expected accuracy |
|----------------|------------------|
| 3 min per class | 80–88% |
| 10 min per class | 88–93% |
| 30 min per class, varied | 93–97% |

---

## API reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/state` | GET | Live detection state |
| `/api/history?hours=24` | GET | Event history |
| `/api/hourly` | GET | Hourly occupancy data |
| `/api/networks` | GET | Per-network RSSI + Kalman stats |
| `/api/summary` | GET | 24h summary (scans, presence time, peak hour) |
| `/api/export/csv?hours=24` | GET | Download events as CSV |
| `/api/config` | GET | Current configuration |
| `/docs` | GET | Interactive API docs (Swagger) |

---

## Configuration

Edit these constants at the top of `backend/main.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SCAN_INTERVAL` | `2.0` | Seconds between WiFi scans |
| `WINDOW_SIZE` | `20` | Rolling window for variance calculation |
| `MOTION_THRESHOLD` | `3.0` | Variance threshold for motion detection |
| `PRESENCE_TIMEOUT` | `25` | Seconds of inactivity before marking absent |

---

## Limitations

- **Requires multiple visible networks** — works best with 5+ visible APs. Single-network environments (e.g. mobile hotspot only) reduce accuracy significantly.
- **Cannot count persons** — WiFi RSSI is aggregate; distinguishing multiple people requires CSI hardware (ESP32-S3).
- **Stationary presence** — a completely still person may not trigger motion detection. The ML model partially addresses this through baseline pattern learning.
- **Environment-specific** — the trained model is specific to your room. Retrain if you move to a different location.
- **Not a security system** — designed for occupancy awareness, not intrusion detection.

---

## Upgrade path

If you want to go beyond WiFi RSSI:

| Hardware | Cost | Capability gained |
|----------|------|------------------|
| ESP32-S3 | ~$8 | Real CSI data, person count, breathing detection |
| PIR sensor + Arduino | ~$5 | Near-perfect motion, zero false positives |
| USB mmWave radar | ~$15 | Detects breathing, works when completely still |

---

## License

MIT
