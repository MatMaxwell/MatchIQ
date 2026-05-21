# ── XGBOOST MODEL TRAINING (LOCAL) ────────────────────────────────────────
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_DIR = os.path.join(BASE_DIR, "data", "gold")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── LOAD GOLD TABLE ────────────────────────────────────────────────────────
df = pd.read_parquet(os.path.join(GOLD_DIR, "speed_dating_features.parquet"))
print(f"Gold table loaded: {len(df)} rows, {len(df.columns)} columns")

# ── DEFINE FEATURES AND TARGET ─────────────────────────────────────────────
feature_cols = [
    "gender", "age_gap", "samerace",
    "attr_delta", "sinc_delta", "intel_delta",
    "fun_delta", "amb_delta", "shar_delta",
    "compatibility_score", "mutual_like",
    "mutual_confidence", "int_corr"
]

X = df[feature_cols].fillna(0)
y = df["match"]

print(f"Features: {len(feature_cols)}")
print(f"Match rate: {y.mean():.2%}")

# ── TRAIN TEST SPLIT ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train: {len(X_train)} rows")
print(f"Test: {len(X_test)} rows")

# ── HANDLE CLASS IMBALANCE ─────────────────────────────────────────────────
# 16% match rate means we need to weight matches higher
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"Class weight ratio: {scale_pos_weight:.2f}")

# ── TRAIN XGBOOST ──────────────────────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    random_state=42,
    verbosity=0
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print("✓ Model trained")

# ── EVALUATE ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n{'='*50}")
print(f"MATCHIQ MODEL RESULTS")
print(f"{'='*50}")
print(f"Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
print(f"AUC Score: {auc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Match", "Match"]))
print(f"Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── FEATURE IMPORTANCE ─────────────────────────────────────────────────────
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print(f"\nFEATURE IMPORTANCE:")
print("="*45)
for _, row in importance_df.iterrows():
    bar = "█" * int(row["importance"] * 100)
    print(f"{row['feature']:25s} {row['importance']:.4f} {bar}")

# ── SAVE MODEL ─────────────────────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, "xgboost_model.json")
model.save_model(model_path)
print(f"\n✓ Model saved to {model_path}")

# Save config
config = {
    "model_path": model_path,
    "feature_cols": feature_cols,
    "version": "1.0.0",
    "auc": round(auc, 4),
    "accuracy": round(accuracy, 4)
}

config_path = os.path.join(MODEL_DIR, "model_config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"✓ Model config saved to {config_path}")
print("\n✓ Model training complete")