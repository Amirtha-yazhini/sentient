#!/usr/bin/env python3
"""
Phase 1 — Data Recorder
Records RSSI snapshots and labels them as 'empty' or 'occupied'.
Run this twice:
  python record.py empty     ← leave the room, run for 2-3 minutes
  python record.py occupied  ← sit/walk in the room, run for 2-3 minutes
"""

import subprocess
import platform
import statistics
import time
import csv
import os
import sys
from datetime import datetime

SCAN_INTERVAL = 1.5   # seconds between scans
OUTPUT_FILE   = "training_data.csv"
DURATION      = 180   # seconds to record (3 minutes default)

# ── WiFi scan (same as backend) ───────────────────────────────────────────
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
                    signal = int(parts[1])
                    freq   = parts[2]
                    bssid  = ":".join(parts[3:]).strip()
                    band   = "5" if freq.startswith("5") else "2"
                    networks.append({"signal": signal, "band": band, "bssid": bssid})
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
            capture_output=True, text=True, timeout=10,
            encoding="utf-8", errors="ignore"
        )
        current = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("SSID") and "BSSID" not in line:
                current = {"ssid": line.split(":", 1)[-1].strip()}
            elif "BSSID" in line and "SSID" not in line:
                current["bssid"] = line.split(":", 1)[-1].strip().replace(" ", "")
            elif "Signal" in line:
                try:
                    current["signal"] = int(line.split(":")[-1].strip().replace("%", ""))
                except:
                    pass
            elif "Radio type" in line:
                current["band"] = "5" if "5" in line else "2"
                if "signal" in current:
                    if "bssid" not in current:
                        current["bssid"] = current.get("ssid", f"net_{len(networks)}")
                    networks.append(dict(current))
                current = {}
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
            if len(parts) >= 3:
                try:
                    rssi = int(parts[-6]) if len(parts) >= 6 else int(parts[1])
                    if rssi < 0:
                        rssi = max(0, min(100, 2 * (rssi + 100)))
                    networks.append({"signal": rssi, "band": "2", "bssid": parts[1]})
                except:
                    pass
    except:
        pass
    return networks

def scan_wifi():
    sys_name = platform.system()
    if sys_name == "Linux":   return scan_wifi_linux()
    if sys_name == "Darwin":  return scan_wifi_mac()
    if sys_name == "Windows": return scan_wifi_windows()
    return []

# ── Feature extraction from one scan snapshot ─────────────────────────────
def extract_features(networks, history):
    """
    Extract a feature vector from current scan + rolling history.
    Features:
      - mean_signal        : average RSSI across all networks
      - max_signal         : strongest network
      - min_signal         : weakest network
      - std_signal         : spread across networks
      - num_networks       : count of visible APs
      - num_5ghz           : count of 5GHz networks
      - num_24ghz          : count of 2.4GHz networks
      - range_signal       : max - min
      - mean_delta         : mean change from last scan
      - std_delta          : spread of changes from last scan
      - rolling_variance   : stdev of mean_signal over last 10 scans
      - rolling_mean_delta : mean of absolute deltas over last 5 scans
    """
    if not networks:
        return None

    signals = [n["signal"] for n in networks]
    mean_s  = statistics.mean(signals)
    max_s   = max(signals)
    min_s   = min(signals)
    std_s   = statistics.stdev(signals) if len(signals) > 1 else 0.0
    n_total = len(signals)
    n_5ghz  = sum(1 for n in networks if str(n.get("band","2")) == "5")
    n_24ghz = n_total - n_5ghz
    range_s = max_s - min_s

    # delta features (change from last snapshot)
    mean_delta = 0.0
    std_delta  = 0.0
    if history:
        last_signals = history[-1]["signals"]
        common = {n.get("bssid", n.get("ssid", "")): n["signal"] for n in networks}
        deltas = []
        for n in last_signals:
            key = n.get("bssid", n.get("ssid", ""))
            if key and key in common:
                deltas.append(abs(common[key] - n["signal"]))
        if deltas:
            mean_delta = statistics.mean(deltas)
            std_delta  = statistics.stdev(deltas) if len(deltas) > 1 else 0.0

    # rolling variance of mean signal (last 10)
    rolling_variance = 0.0
    if len(history) >= 3:
        recent_means = [h["mean_signal"] for h in history[-10:]] + [mean_s]
        rolling_variance = statistics.stdev(recent_means) if len(recent_means) > 1 else 0.0

    # rolling mean absolute delta (last 5)
    rolling_mean_delta = 0.0
    if len(history) >= 2:
        recent_deltas = [h.get("mean_delta", 0) for h in history[-5:]] + [mean_delta]
        rolling_mean_delta = statistics.mean(recent_deltas)

    return {
        "mean_signal":        round(mean_s, 2),
        "max_signal":         max_s,
        "min_signal":         min_s,
        "std_signal":         round(std_s, 2),
        "num_networks":       n_total,
        "num_5ghz":           n_5ghz,
        "num_24ghz":          n_24ghz,
        "range_signal":       range_s,
        "mean_delta":         round(mean_delta, 2),
        "std_delta":          round(std_delta, 2),
        "rolling_variance":   round(rolling_variance, 2),
        "rolling_mean_delta": round(rolling_mean_delta, 2),
    }

# ── Main recorder ─────────────────────────────────────────────────────────
def record(label):
    print(f"\n=== Recording label: '{label}' ===")
    if label == "empty":
        print(">> Leave the room now. Recording starts in 5 seconds...")
    else:
        print(">> Stay in the room, move around normally. Recording starts in 5 seconds...")

    for i in range(5, 0, -1):
        print(f"   {i}...", end="\r")
        time.sleep(1)
    print(f"\nRecording for {DURATION} seconds. Press Ctrl+C to stop early.\n")

    history  = []
    rows     = []
    start    = time.time()
    scan_num = 0

    # check if file exists to decide whether to write header
    file_exists = os.path.isfile(OUTPUT_FILE)

    FEATURE_COLS = [
        "mean_signal","max_signal","min_signal","std_signal",
        "num_networks","num_5ghz","num_24ghz","range_signal",
        "mean_delta","std_delta","rolling_variance","rolling_mean_delta",
        "label"
    ]

    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLS)
        if not file_exists:
            writer.writeheader()

        try:
            while time.time() - start < DURATION:
                networks = scan_wifi()
                elapsed  = time.time() - start
                scan_num += 1

                if not networks:
                    print(f"  [{elapsed:.0f}s] No networks found, retrying...")
                    time.sleep(SCAN_INTERVAL)
                    continue

                features = extract_features(networks, history)
                if features:
                    history.append({
                        "signals":     networks,
                        "mean_signal": features["mean_signal"],
                        "mean_delta":  features["mean_delta"],
                    })
                    row = {**features, "label": label}
                    writer.writerow(row)
                    f.flush()
                    rows.append(row)

                    bar = "█" * int((elapsed / DURATION) * 30)
                    bar = bar.ljust(30)
                    print(f"  [{elapsed:5.1f}s] [{bar}] scan #{scan_num} | mean={features['mean_signal']:.1f} | Δ={features['mean_delta']:.2f} | var={features['rolling_variance']:.2f}    ", end="\r")

                time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            print("\n\nStopped early.")

    print(f"\n\nDone. Recorded {len(rows)} samples → {OUTPUT_FILE}")
    return rows

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("empty", "occupied"):
        print("Usage:")
        print("  python record.py empty      ← leave the room first")
        print("  python record.py occupied   ← stay/move in the room")
        sys.exit(1)

    label = sys.argv[1]
    record(label)
    print(f"\nNext step: run 'python train.py' once you have both labels recorded.")