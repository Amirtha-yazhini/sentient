#!/usr/bin/env python3
"""
Phase 2 — Train Classifier
Reads training_data.csv and trains a Random Forest classifier.
Saves model to model.pkl for use by the backend.

Run: python train.py
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline         import Pipeline

DATA_FILE  = "training_data.csv"
MODEL_FILE = "model.pkl"
META_FILE  = "model_meta.json"

FEATURE_COLS = [
    "mean_signal", "max_signal", "min_signal", "std_signal",
    "num_networks", "num_5ghz", "num_24ghz", "range_signal",
    "mean_delta", "std_delta", "rolling_variance", "rolling_mean_delta",
]

def load_data():
    if not os.path.isfile(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found.")
        print("Run record.py first to collect training data.")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE)
    print(f"\nLoaded {len(df)} samples from {DATA_FILE}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

    counts = df["label"].value_counts()
    if len(counts) < 2:
        print("ERROR: Need both 'empty' and 'occupied' labels in training data.")
        print("Run: python record.py empty")
        print("Then: python record.py occupied")
        sys.exit(1)

    min_count = counts.min()
    if min_count < 10:
        print(f"WARNING: Only {min_count} samples for one class. Collect more data for better accuracy.")

    return df

def train(df):
    X = df[FEATURE_COLS].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest classifier...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ))
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"\nCross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Final fit
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f} ({acc*100:.1f}%)\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=["empty", "occupied"])
    print(f"              Predicted")
    print(f"              empty  occupied")
    print(f"Actual empty  {cm[0][0]:5d}  {cm[0][1]:8d}")
    print(f"     occupied {cm[1][0]:5d}  {cm[1][1]:8d}")

    # Feature importance
    rf = pipeline.named_steps["clf"]
    importances = sorted(zip(FEATURE_COLS, rf.feature_importances_), key=lambda x: -x[1])
    print("\nFeature importances:")
    for feat, imp in importances:
        bar = "█" * int(imp * 40)
        print(f"  {feat:25s} {imp:.3f}  {bar}")

    return pipeline, acc, cv_scores.mean()

def save_model(pipeline, acc, cv_acc):
    joblib.dump(pipeline, MODEL_FILE)
    print(f"\nModel saved → {MODEL_FILE}")

    meta = {
        "features":     FEATURE_COLS,
        "classes":      list(pipeline.classes_),
        "test_accuracy": round(float(acc), 4),
        "cv_accuracy":   round(float(cv_acc), 4),
        "trained_at":    pd.Timestamp.now().isoformat(),
        "n_estimators":  200,
        "model_type":   "RandomForest + StandardScaler",
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved → {META_FILE}")
    print(f"\nAccuracy: {acc*100:.1f}%")

    if acc >= 0.90:
        print("✓ Excellent! Model is ready to use.")
    elif acc >= 0.80:
        print("✓ Good. Model is usable. Collect more data to improve.")
    else:
        print("⚠ Accuracy below 80%. Collect more varied data and retrain.")

if __name__ == "__main__":
    print("=== Presence Detector — Training ===")
    df       = load_data()
    pipeline, acc, cv_acc = train(df)
    save_model(pipeline, acc, cv_acc)
    print("\nNext step: restart main.py — it will auto-load the model.")
