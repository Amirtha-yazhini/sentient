"""
Presence Detector Backend — FastAPI
Features:
  - WiFi RSSI scanning (Linux/macOS/Windows)
  - Per-network RSSI tracking (2.4GHz + 5GHz)
  - Kalman filter noise reduction
  - Activity classification (absent/still/active/walking)
  - Confidence score
  - SQLite persistence + history
  - Webcam person count (OpenCV HOG)
  - Desktop notifications
  - CSV export
  - REST API
"""

import asyncio
import json
import math
import os
import platform
import statistics
import subprocess
import sqlite3
import time
import threading
import csv
import io
from collections import deque
from datetime import datetime, timedelta
from typing import Optional, List

import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── ML model (optional — loads if model.pkl exists) ──────────────────────
ML_MODEL      = None
ML_META       = None
ML_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "model.pkl")
ML_META_PATH  = os.path.join(os.path.dirname(__file__), "..", "ml", "model_meta.json")

def try_load_model():
    global ML_MODEL, ML_META
    try:
        import joblib
        if os.path.isfile(ML_MODEL_PATH):
            ML_MODEL = joblib.load(ML_MODEL_PATH)
            if os.path.isfile(ML_META_PATH):
                with open(ML_META_PATH) as f:
                    ML_META = json.load(f)
            acc = ML_META.get("test_accuracy", 0) if ML_META else 0
            print(f"[ML] Model loaded — accuracy {acc*100:.1f}%")
            return True
    except Exception as e:
        print(f"[ML] Could not load model: {e}")
    return False

ML_FEATURE_COLS = [
    "mean_signal", "max_signal", "min_signal", "std_signal",
    "num_networks", "num_5ghz", "num_24ghz", "range_signal",
    "mean_delta", "std_delta", "rolling_variance", "rolling_mean_delta",
]

def ml_predict(features: dict):
    """Returns (label, confidence) using ML model if available."""
    if ML_MODEL is None:
        return None, None
    try:
        row = np.array([[features.get(c, 0.0) for c in ML_FEATURE_COLS]])
        pred  = ML_MODEL.predict(row)[0]
        proba = ML_MODEL.predict_proba(row)[0]
        conf  = round(float(max(proba)) * 100, 1)
        return pred, conf
    except Exception as e:
        return None, None

# ── Kalman filter (1D, manual — no external dep needed) ─────────────────────
class KalmanFilter1D:
    def __init__(self, process_noise=0.01, measurement_noise=2.0):
        self.x = 0.0        # state estimate
        self.p = 1.0        # estimate error covariance
        self.q = process_noise
        self.r = measurement_noise

    def update(self, z):
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (z - self.x)
        self.p *= (1 - k)
        return self.x

# ── Config ───────────────────────────────────────────────────────────────────
SCAN_INTERVAL    = 2.0
WINDOW_SIZE      = 20
MOTION_THRESHOLD = 3.0
PRESENCE_TIMEOUT = 25
DB_PATH          = "presence.db"
NOTIFIED         = {"last": 0}   # simple debounce for desktop notif

# ── State ────────────────────────────────────────────────────────────────────
state = {
    "rssi_mean_history": deque(maxlen=WINDOW_SIZE),
    "per_network":       {},          # bssid -> deque of RSSI
    "kalman_filters":    {},          # bssid -> KalmanFilter1D
    "variance_history":  deque(maxlen=120),
    "confidence_history":deque(maxlen=120),
    "last_motion_time":  time.time(),
    "presence":          False,
    "motion":            False,
    "activity":          "absent",    # absent | still | active | walking
    "confidence":        0.0,
    "variance":          0.0,
    "current_networks":  [],
    "person_count":      0,           # from webcam
    "scan_count":        0,
    "status":            "warming_up",
    "log":               deque(maxlen=100),
    "hourly_occupancy":  {},          # "HH": minutes_occupied
    "zone":              "unknown",   # near | mid | far | unknown
    "ml_label":          None,        # ML prediction: empty | occupied
    "ml_confidence":     None,        # ML confidence %
    "using_ml":          False,
    "rssi_snapshot_history": deque(maxlen=20),
}

kalman_mean = KalmanFilter1D(process_noise=0.05, measurement_noise=3.0)

# ── DB ───────────────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts        REAL NOT NULL,
            activity  TEXT,
            confidence REAL,
            variance  REAL,
            person_count INTEGER,
            zone      TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hourly (
            hour TEXT PRIMARY KEY,
            minutes_occupied INTEGER DEFAULT 0
        )
    """)
    con.commit()
    con.close()

def db_insert(activity, confidence, variance, person_count, zone):
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT INTO events(ts,activity,confidence,variance,person_count,zone) VALUES(?,?,?,?,?,?)",
            (time.time(), activity, round(confidence,2), round(variance,2), person_count, zone)
        )
        hour = datetime.now().strftime("%Y-%m-%d %H:00")
        if activity != "absent":
            con.execute(
                "INSERT INTO hourly(hour,minutes_occupied) VALUES(?,?) "
                "ON CONFLICT(hour) DO UPDATE SET minutes_occupied=minutes_occupied+?",
                (hour, round(SCAN_INTERVAL/60, 2), round(SCAN_INTERVAL/60, 2))
            )
        con.commit()
        con.close()
    except Exception as e:
        pass

def db_history(hours=24):
    try:
        since = time.time() - hours * 3600
        con = sqlite3.connect(DB_PATH)
        rows = con.execute(
            "SELECT ts,activity,confidence,variance,person_count,zone FROM events "
            "WHERE ts > ? ORDER BY ts DESC LIMIT 500", (since,)
        ).fetchall()
        con.close()
        return [{"ts": r[0], "activity": r[1], "confidence": r[2],
                 "variance": r[3], "person_count": r[4], "zone": r[5]} for r in rows]
    except:
        return []

def db_hourly():
    try:
        since = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d %H:00")
        con = sqlite3.connect(DB_PATH)
        rows = con.execute(
            "SELECT hour, minutes_occupied FROM hourly WHERE hour >= ? ORDER BY hour",
            (since,)
        ).fetchall()
        con.close()
        return [{"hour": r[0], "minutes": round(r[1],1)} for r in rows]
    except:
        return []

# ── WiFi scan ────────────────────────────────────────────────────────────────
def scan_wifi_linux():
    networks = []
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "SSID,SIGNAL,FREQ,BSSID", "dev", "wifi", "list", "--rescan", "yes"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split(":")
            if len(parts) >= 4:
                try:
                    ssid   = parts[0] or "hidden"
                    signal = int(parts[1])
                    freq   = parts[2]
                    bssid  = ":".join(parts[3:]).strip()
                    band   = "5GHz" if freq.startswith("5") else "2.4GHz"
                    networks.append({"ssid": ssid, "signal": signal, "band": band, "bssid": bssid})
                except:
                    pass
    except:
        pass
    return networks

def scan_wifi_mac():
    networks = []
    airport = "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport"
    try:
        result = subprocess.run([airport, "-s"], capture_output=True, text=True, timeout=10)
        for line in result.stdout.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 7:
                try:
                    rssi = int(parts[-6])
                    if rssi < 0:
                        rssi = max(0, min(100, 2 * (rssi + 100)))
                    networks.append({"ssid": parts[0], "signal": rssi, "band": "2.4GHz", "bssid": parts[1]})
                except:
                    pass
    except:
        pass
    return networks

def scan_wifi_windows():
    networks = []
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "networks", "mode=bssid"],
            capture_output=True, text=True, timeout=10, encoding="utf-8", errors="ignore"
        )
        current = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("SSID") and "BSSID" not in line:
                current["ssid"] = line.split(":", 1)[-1].strip()
            elif "BSSID" in line:
                current["bssid"] = line.split(":", 1)[-1].strip()
            elif "Signal" in line:
                try:
                    current["signal"] = int(line.split(":")[-1].strip().replace("%", ""))
                except:
                    pass
            elif "Radio type" in line:
                current["band"] = "5GHz" if "5" in line else "2.4GHz"
                networks.append({**current})
                current = {}
    except:
        pass
    return networks

def scan_wifi():
    sys = platform.system()
    if sys == "Linux":   return scan_wifi_linux()
    if sys == "Darwin":  return scan_wifi_mac()
    if sys == "Windows": return scan_wifi_windows()
    return []

# ── Zone estimation ──────────────────────────────────────────────────────────
def estimate_zone(networks):
    if not networks:
        return "unknown"
    best = max(n["signal"] for n in networks)
    if best >= 70:   return "near"
    if best >= 45:   return "mid"
    return "far"

# ── Activity classification ──────────────────────────────────────────────────
def classify_activity(variance, confidence):
    if variance < 1.0:                        return "absent"
    if variance < MOTION_THRESHOLD:           return "still"
    if variance < MOTION_THRESHOLD * 2.0:     return "active"
    return "walking"

# ── Desktop notification ──────────────────────────────────────────────────────
def notify(title, msg):
    now = time.time()
    if now - NOTIFIED["last"] < 30:
        return
    NOTIFIED["last"] = now
    try:
        sys = platform.system()
        if sys == "Darwin":
            subprocess.Popen(["osascript", "-e", f'display notification "{msg}" with title "{title}"'])
        elif sys == "Linux":
            subprocess.Popen(["notify-send", title, msg])
        elif sys == "Windows":
            # Use plyer if available, else silent
            try:
                from plyer import notification
                notification.notify(title=title, message=msg, app_name="Sentient", timeout=5)
            except ImportError:
                pass  # notifications disabled — run: pip install plyer
    except:
        pass

# ── Webcam person count ───────────────────────────────────────────────────────
_cam_lock = threading.Lock()
_last_person_count = 0

def webcam_count():
    global _last_person_count
    try:
        import cv2
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        with _cam_lock:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return _last_person_count
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return _last_person_count
            frame = cv2.resize(frame, (640, 480))
            rects, _ = hog.detectMultiScale(frame, winStride=(8,8), padding=(4,4), scale=1.05)
            _last_person_count = len(rects)
            return _last_person_count
    except Exception:
        return _last_person_count

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    state["log"].appendleft(entry)
    print(entry)

# ── ML feature extractor (mirrors record.py) ─────────────────────────────
def extract_ml_features(networks, snap_history):
    if not networks:
        return None
    signals  = [n["signal"] for n in networks]
    mean_s   = statistics.mean(signals)
    max_s    = max(signals)
    min_s    = min(signals)
    std_s    = statistics.stdev(signals) if len(signals) > 1 else 0.0
    n_total  = len(signals)
    n_5ghz   = sum(1 for n in networks if str(n.get("band","")).startswith("5"))
    range_s  = max_s - min_s

    mean_delta = 0.0
    std_delta  = 0.0
    if snap_history:
        last = snap_history[-1]
        common_map = {n.get("bssid",""): n["signal"] for n in networks}
        deltas = [abs(common_map[n["bssid"]] - n["signal"])
                  for n in last["networks"] if n.get("bssid") in common_map]
        if deltas:
            mean_delta = statistics.mean(deltas)
            std_delta  = statistics.stdev(deltas) if len(deltas) > 1 else 0.0

    rolling_variance = 0.0
    if len(snap_history) >= 3:
        recent = [h["mean_signal"] for h in snap_history[-10:]] + [mean_s]
        rolling_variance = statistics.stdev(recent) if len(recent) > 1 else 0.0

    rolling_mean_delta = 0.0
    if len(snap_history) >= 2:
        recent_d = [h.get("mean_delta", 0) for h in snap_history[-5:]] + [mean_delta]
        rolling_mean_delta = statistics.mean(recent_d)

    return {
        "mean_signal":        round(mean_s, 2),
        "max_signal":         max_s,
        "min_signal":         min_s,
        "std_signal":         round(std_s, 2),
        "num_networks":       n_total,
        "num_5ghz":           n_5ghz,
        "num_24ghz":          n_total - n_5ghz,
        "range_signal":       range_s,
        "mean_delta":         round(mean_delta, 2),
        "std_delta":          round(std_delta, 2),
        "rolling_variance":   round(rolling_variance, 2),
        "rolling_mean_delta": round(rolling_mean_delta, 2),
    }

# ── Main detection loop ───────────────────────────────────────────────────────
def detection_loop():
    log("Detector starting...")
    state["status"] = "warming_up"
    cam_tick = 0

    while True:
        try:
            networks = scan_wifi()
            state["scan_count"] += 1

            if not networks:
                state["status"] = "error_no_wifi"
                time.sleep(SCAN_INTERVAL)
                continue

            # Per-network kalman-filtered RSSI
            for n in networks:
                bssid = n["bssid"]
                if bssid not in state["per_network"]:
                    state["per_network"][bssid] = deque(maxlen=WINDOW_SIZE)
                    state["kalman_filters"][bssid] = KalmanFilter1D()
                kf = state["kalman_filters"][bssid]
                filtered = kf.update(n["signal"])
                state["per_network"][bssid].append(filtered)

            # Mean across all networks, kalman-filtered
            raw_mean = statistics.mean(n["signal"] for n in networks)
            smooth_mean = kalman_mean.update(raw_mean)
            state["rssi_mean_history"].append(smooth_mean)
            state["current_networks"] = networks[:15]
            state["zone"] = estimate_zone(networks)

            if len(state["rssi_mean_history"]) < 5:
                state["status"] = "warming_up"
                time.sleep(SCAN_INTERVAL)
                continue

            # Variance = motion signal
            var = statistics.stdev(list(state["rssi_mean_history"]))
            state["variance"] = round(var, 2)
            state["variance_history"].append(round(var, 2))

            # Confidence: how far above/below threshold (0-100)
            conf = min(100.0, max(0.0, (var / (MOTION_THRESHOLD * 2)) * 100))
            state["confidence"] = round(conf, 1)
            state["confidence_history"].append(round(conf, 1))

            motion  = var > MOTION_THRESHOLD
            state["motion"] = motion
            activity = classify_activity(var, conf)
            state["activity"] = activity

            # ── ML prediction (overrides activity if model loaded) ────────
            ml_features = extract_ml_features(networks, list(state["rssi_snapshot_history"]))
            if ml_features:
                state["rssi_snapshot_history"].append({
                    "networks":    networks,
                    "mean_signal": ml_features["mean_signal"],
                    "mean_delta":  ml_features["mean_delta"],
                })
                ml_label, ml_conf = ml_predict(ml_features)
                if ml_label is not None:
                    state["ml_label"]      = ml_label
                    state["ml_confidence"] = ml_conf
                    state["using_ml"]      = True
                    # use ML to override presence decision
                    if ml_label == "occupied":
                        motion = True
                        state["motion"]     = True
                        state["confidence"] = ml_conf
                        if activity == "absent":
                            activity = "still"
                            state["activity"] = "still"
                    else:
                        # ML says empty — only override if variance also low
                        if var < MOTION_THRESHOLD:
                            motion = False
                            state["motion"]   = False
                            activity = "absent"
                            state["activity"] = "absent"
                            state["confidence"] = round(100 - ml_conf, 1)

            # Webcam every 5 scans (~10s)
            cam_tick += 1
            if cam_tick >= 5:
                cam_tick = 0
                state["person_count"] = webcam_count()

            # Presence logic
            if motion:
                state["last_motion_time"] = time.time()
                if not state["presence"]:
                    state["presence"] = True
                    log(f"PRESENCE DETECTED — {activity} (var={var:.2f}, conf={conf:.0f}%)")
                    notify("Presence Detected", f"Activity: {activity}")
            else:
                idle = time.time() - state["last_motion_time"]
                if idle > PRESENCE_TIMEOUT and state["presence"]:
                    state["presence"] = False
                    log(f"Area cleared (idle {idle:.0f}s)")
                    notify("Area Clear", "No presence detected")

            state["status"] = activity

            # Persist every 5 scans
            if state["scan_count"] % 5 == 0:
                db_insert(activity, conf, var, state["person_count"], state["zone"])

        except Exception as e:
            log(f"Error: {e}")

        time.sleep(SCAN_INTERVAL)

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Presence Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    init_db()
    try_load_model()
    state["using_ml"] = ML_MODEL is not None
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/state")
def get_state():
    return {
        "status":            state["status"],
        "presence":          state["presence"],
        "motion":            state["motion"],
        "activity":          state["activity"],
        "confidence":        state["confidence"],
        "variance":          state["variance"],
        "zone":              state["zone"],
        "person_count":      state["person_count"],
        "scan_count":        state["scan_count"],
        "variance_history":  list(state["variance_history"]),
        "confidence_history":list(state["confidence_history"]),
        "current_networks":  state["current_networks"],
        "log":               list(state["log"])[:30],
        "threshold":         MOTION_THRESHOLD,
        "scan_interval":     SCAN_INTERVAL,
        "ml_label":          state["ml_label"],
        "ml_confidence":     state["ml_confidence"],
        "using_ml":          state["using_ml"],
        "ml_accuracy":       ML_META.get("test_accuracy") if ML_META else None,
    }

@app.get("/api/history")
def get_history(hours: int = Query(24, ge=1, le=168)):
    return {"events": db_history(hours)}

@app.get("/api/hourly")
def get_hourly():
    return {"data": db_hourly()}

@app.get("/api/networks")
def get_networks():
    per = {}
    for bssid, vals in state["per_network"].items():
        if vals:
            per[bssid] = {
                "current": round(list(vals)[-1], 1),
                "mean":    round(statistics.mean(vals), 1),
                "stdev":   round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
                "samples": len(vals),
            }
    return {"networks": state["current_networks"], "per_bssid": per}

@app.get("/api/export/csv")
def export_csv(hours: int = Query(24, ge=1, le=168)):
    events = db_history(hours)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["ts","datetime","activity","confidence","variance","person_count","zone"])
    writer.writeheader()
    for e in events:
        e["datetime"] = datetime.fromtimestamp(e["ts"]).strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow(e)
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=presence_{hours}h.csv"}
    )

@app.get("/api/config")
def get_config():
    return {
        "threshold":       MOTION_THRESHOLD,
        "scan_interval":   SCAN_INTERVAL,
        "presence_timeout":PRESENCE_TIMEOUT,
        "window_size":     WINDOW_SIZE,
        "platform":        platform.system(),
    }

@app.get("/api/summary")
def get_summary():
    events = db_history(24)
    if not events:
        return {"total_scans": state["scan_count"], "presence_minutes": 0, "peak_hour": "N/A", "avg_confidence": 0}
    present = [e for e in events if e["activity"] != "absent"]
    presence_mins = round(len(present) * SCAN_INTERVAL / 60, 1)
    by_hour = {}
    for e in present:
        h = datetime.fromtimestamp(e["ts"]).strftime("%H:00")
        by_hour[h] = by_hour.get(h, 0) + 1
    peak = max(by_hour, key=by_hour.get) if by_hour else "N/A"
    avg_conf = round(statistics.mean(e["confidence"] for e in present), 1) if present else 0
    return {
        "total_scans":     state["scan_count"],
        "presence_minutes":presence_mins,
        "peak_hour":       peak,
        "avg_confidence":  avg_conf,
        "events_24h":      len(events),
    }

if __name__ == "__main__":
    import uvicorn
    print("Backend → http://localhost:8000")
    print("API docs → http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)