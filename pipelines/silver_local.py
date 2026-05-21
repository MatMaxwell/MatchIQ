# ── SILVER CLEANING (LOCAL) ────────────────────────────────────────────────
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRONZE_DIR = os.path.join(BASE_DIR, "data", "bronze")
SILVER_DIR = os.path.join(BASE_DIR, "data", "silver")

os.makedirs(os.path.join(SILVER_DIR, "speed_dating"), exist_ok=True)
os.makedirs(os.path.join(SILVER_DIR, "okcupid"), exist_ok=True)

# ── LOAD BRONZE ────────────────────────────────────────────────────────────
speed_df = pd.read_parquet(os.path.join(BRONZE_DIR, "speed_dating", "speed_dating.parquet"))
okcupid_df = pd.read_parquet(os.path.join(BRONZE_DIR, "okcupid", "okcupid.parquet"))

print(f"Bronze Speed Dating loaded: {len(speed_df)} rows")
print(f"Bronze OKCupid loaded: {len(okcupid_df)} rows")

# ── CLEAN SPEED DATING ─────────────────────────────────────────────────────

# Step 1: Deduplicate — one row per unique encounter
speed_df = speed_df.drop_duplicates(subset=["iid", "pid"])
print(f"After dedup: {len(speed_df)} rows")

# Step 2: Fix data types
numeric_cols = ["age", "age_o", "match", "attr", "sinc", "intel",
                "fun", "amb", "shar", "attr_o", "sinc_o", "intel_o",
                "fun_o", "amb_o", "shar_o", "like", "like_o", "prob", "prob_o", "int_corr"]

for col in numeric_cols:
    if col in speed_df.columns:
        speed_df[col] = pd.to_numeric(speed_df[col], errors="coerce")

print("✓ Data types fixed")

# Step 3: Impute missing values with column means
rating_cols = ["attr", "sinc", "intel", "fun", "amb", "shar",
               "attr_o", "sinc_o", "intel_o", "fun_o", "amb_o", "shar_o"]

for col in rating_cols:
    if col in speed_df.columns:
        speed_df[col] = speed_df[col].fillna(speed_df[col].mean())

print("✓ Missing values imputed")

# Step 4: Remove rows with no match label
speed_df = speed_df[speed_df["match"].notna()]
print(f"After removing null match labels: {len(speed_df)} rows")

# Step 5: Select relevant columns
keep_cols = ["iid", "pid", "match", "gender", "age", "age_o",
             "race", "race_o", "samerace", "int_corr",
             "attr", "sinc", "intel", "fun", "amb", "shar",
             "attr_o", "sinc_o", "intel_o", "fun_o", "amb_o", "shar_o",
             "like", "like_o", "prob", "prob_o"]

keep_cols = [c for c in keep_cols if c in speed_df.columns]
speed_df = speed_df[keep_cols]
print(f"✓ Selected {len(speed_df.columns)} columns")



# ── WRITE SILVER ───────────────────────────────────────────────────────────
speed_df.to_parquet(os.path.join(SILVER_DIR, "speed_dating", "speed_dating.parquet"), index=False)
print(f"\n✓ Silver Speed Dating: {len(speed_df)} rows, {len(speed_df.columns)} columns")
print("✓ Silver cleaning complete")