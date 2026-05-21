# ── BRONZE INGESTION (LOCAL) ───────────────────────────────────────────────
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
BRONZE_DIR = os.path.join(BASE_DIR, "data", "bronze")

os.makedirs(os.path.join(BRONZE_DIR, "speed_dating"), exist_ok=True)
os.makedirs(os.path.join(BRONZE_DIR, "okcupid"), exist_ok=True)

# ── INGEST SPEED DATING ────────────────────────────────────────────────────
speed_path = os.path.join(RAW_DIR, "speed_dating", "Speed Dating Data.csv")
speed_df = pd.read_csv(speed_path, encoding="latin-1")
print(f"✓ Speed Dating loaded: {len(speed_df)} rows, {len(speed_df.columns)} columns")

speed_df.to_parquet(os.path.join(BRONZE_DIR, "speed_dating", "speed_dating.parquet"), index=False)
print("✓ Bronze Speed Dating written")

# ── INGEST OKCUPID ─────────────────────────────────────────────────────────
okcupid_path = os.path.join(RAW_DIR, "okcupid", "parsed_data_public.parquet")
okcupid_df = pd.read_parquet(okcupid_path)
print(f"✓ OKCupid loaded: {len(okcupid_df)} rows, {len(okcupid_df.columns)} columns")

okcupid_df.to_parquet(os.path.join(BRONZE_DIR, "okcupid", "okcupid.parquet"), index=False)
print("✓ Bronze OKCupid written")

print("\n✓ Bronze ingestion complete")