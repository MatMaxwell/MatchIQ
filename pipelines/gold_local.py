# ── GOLD FEATURE ENGINEERING (LOCAL) ──────────────────────────────────────
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILVER_DIR = os.path.join(BASE_DIR, "data", "silver")
GOLD_DIR = os.path.join(BASE_DIR, "data", "gold")

os.makedirs(GOLD_DIR, exist_ok=True)

# ── LOAD SILVER ────────────────────────────────────────────────────────────
df = pd.read_parquet(os.path.join(SILVER_DIR, "speed_dating", "speed_dating.parquet"))
print(f"Silver loaded: {len(df)} rows")

# ── FEATURE 1: MUTUAL RATING DELTAS ───────────────────────────────────────
# How differently did they rate each other on each attribute?
# Smaller gap = more aligned = better compatibility signal

df["attr_delta"]  = abs(df["attr"]  - df["attr_o"])
df["sinc_delta"]  = abs(df["sinc"]  - df["sinc_o"])
df["intel_delta"] = abs(df["intel"] - df["intel_o"])
df["fun_delta"]   = abs(df["fun"]   - df["fun_o"])
df["amb_delta"]   = abs(df["amb"]   - df["amb_o"])
df["shar_delta"]  = abs(df["shar"]  - df["shar_o"])

print("✓ Mutual rating deltas computed")

# ── FEATURE 2: OVERALL COMPATIBILITY SCORE ─────────────────────────────────
# Average of all 6 deltas — one summary compatibility number
# Lower = more compatible

df["compatibility_score"] = df[["attr_delta", "sinc_delta", "intel_delta",
                                  "fun_delta", "amb_delta", "shar_delta"]].mean(axis=1).round(2)

print("✓ Compatibility score computed")

# ── FEATURE 3: MUTUAL LIKE SCORE ──────────────────────────────────────────
# Average of how much each person liked the other overall

df["mutual_like"] = ((df["like"] + df["like_o"]) / 2).round(2)

print("✓ Mutual like score computed")

# ── FEATURE 4: MUTUAL CONFIDENCE ──────────────────────────────────────────
# Average of how confident each person was the other would say yes

df["mutual_confidence"] = ((df["prob"] + df["prob_o"]) / 2).round(2)

print("✓ Mutual confidence computed")

# ── FEATURE 5: AGE GAP ────────────────────────────────────────────────────
# Absolute age difference between the pair

df["age_gap"] = abs(df["age"] - df["age_o"])

print("✓ Age gap computed")

# ── SELECT FINAL GOLD FEATURES ─────────────────────────────────────────────
feature_cols = [
    "iid", "pid",           # identifiers
    "match",                # TARGET LABEL
    "gender",               # demographics
    "age_gap",
    "samerace",
    "attr_delta",           # compatibility deltas
    "sinc_delta",
    "intel_delta",
    "fun_delta",
    "amb_delta",
    "shar_delta",
    "compatibility_score",  # aggregate scores
    "mutual_like",
    "mutual_confidence",
    "int_corr"              # interest correlation
]

# Only keep columns that exist
feature_cols = [c for c in feature_cols if c in df.columns]
gold_df = df[feature_cols].copy()

# Fill any remaining nulls with 0
gold_df = gold_df.fillna(0)

print(f"\n✓ Gold feature set: {len(gold_df.columns)} columns")
print(f"✓ Training rows: {len(gold_df)}")
print(f"\nClass distribution:")
print(gold_df["match"].value_counts())
print(f"Match rate: {gold_df['match'].mean():.2%}")

# ── WRITE GOLD ─────────────────────────────────────────────────────────────
gold_df.to_parquet(os.path.join(GOLD_DIR, "speed_dating_features.parquet"), index=False)
print(f"\n✓ Gold table written to {GOLD_DIR}")
print("✓ Gold feature engineering complete")
