# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 03 - Gold Feature Engineering
# MAGIC
# MAGIC **What is Gold?**
# MAGIC Gold is the final layer of the Medallion Architecture.
# MAGIC This is where we transform clean Silver data into ML-ready features.
# MAGIC Every row in the Gold table will be one training example for XGBoost.
# MAGIC
# MAGIC **What is Feature Engineering?**
# MAGIC Raw data columns are rarely useful directly for ML.
# MAGIC Feature engineering means creating NEW columns that better capture
# MAGIC the patterns we want the model to learn.
# MAGIC Example: instead of feeding the model "person rated attr=8, partner rated attr=4"
# MAGIC we create "attractiveness_delta = 4" which directly captures the gap.
# MAGIC The model learns from deltas much better than raw numbers.
# MAGIC
# MAGIC **What we build here:**
# MAGIC 1. Compatibility deltas (gaps between mutual ratings)
# MAGIC 2. Agreement scores (how aligned their preferences are)
# MAGIC 3. demographic similarity features
# MAGIC 4. Final ML training table with target label

# COMMAND ----------

# ── IMPORTS ────────────────────────────────────────────────────────────────
from pyspark.sql.functions import (
    col, abs, when, isnull, mean,
    sqrt, pow, lit, round,
    monotonically_increasing_id
)
from pyspark.sql.types import FloatType, IntegerType

# COMMAND ----------

# ── STORAGE CONFIGURATION ──────────────────────────────────────────────────
storage_account = "matchiqstorage"
container = "matchiq"

spark.conf.set(
    f"fs.azure.account.auth.type.{storage_account}.dfs.core.windows.net",
    "OAuth"
)
spark.conf.set(
    f"fs.azure.account.oauth.provider.type.{storage_account}.dfs.core.windows.net",
    "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider"
)

base_path = f"abfss://{container}@{storage_account}.dfs.core.windows.net"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Silver Tables

# COMMAND ----------

# ── READ FROM SILVER ───────────────────────────────────────────────────────
# We always read from the previous layer
# Silver is clean and typed correctly — safe to engineer features from

speed_df = spark.read.format("delta") \
    .load(f"{base_path}/silver/speed_dating")

print(f"Silver Speed Dating loaded: {speed_df.count()} rows")
speed_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering: Compatibility Deltas
# MAGIC
# MAGIC A delta is just the absolute difference between two values.
# MAGIC Think of it like this — if Person A rates Person B's attractiveness as 8
# MAGIC but Person B rates Person A's attractiveness as 3, the delta is 5.
# MAGIC A large delta = they see each other very differently = likely incompatible.
# MAGIC A small delta = they see each other similarly = likely compatible.
# MAGIC
# MAGIC We compute deltas for all 6 rating attributes.
# MAGIC abs() = absolute value so the delta is always positive
# MAGIC (we care about the SIZE of the gap, not the direction)

# COMMAND ----------

# ── MUTUAL RATING DELTAS ───────────────────────────────────────────────────
# For each of the 6 attributes, compute how differently they rated each other
# attr = how person rated partner's attractiveness
# attr_o = how partner rated person's attractiveness
# attr_delta = absolute difference between these two ratings

speed_df = speed_df \
    .withColumn("attr_delta",
        abs(col("attr") - col("attr_o"))        # attractiveness gap
    ) \
    .withColumn("sinc_delta",
        abs(col("sinc") - col("sinc_o"))        # sincerity gap
    ) \
    .withColumn("intel_delta",
        abs(col("intel") - col("intel_o"))      # intelligence gap
    ) \
    .withColumn("fun_delta",
        abs(col("fun") - col("fun_o"))          # fun gap
    ) \
    .withColumn("amb_delta",
        abs(col("amb") - col("amb_o"))          # ambition gap
    ) \
    .withColumn("shar_delta",
        abs(col("shar") - col("shar_o"))        # shared interests gap
    )

print("✓ Mutual rating deltas computed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering: Aggregate Compatibility Score
# MAGIC
# MAGIC Instead of feeding the model 6 separate deltas, we also create
# MAGIC one overall compatibility score that summarizes all of them.
# MAGIC Think of it as the average gap across all 6 attributes.
# MAGIC A score close to 0 = very compatible, close to 10 = very incompatible.
# MAGIC
# MAGIC We also compute a MUTUAL LIKE SCORE — the average of how much
# MAGIC each person liked the other overall. This is a strong predictor.

# COMMAND ----------

# ── OVERALL COMPATIBILITY DELTA ────────────────────────────────────────────
# Average of all 6 deltas into one summary score
# Lower = more compatible (smaller gaps across all attributes)

speed_df = speed_df.withColumn(
    "compatibility_score",
    round(
        (col("attr_delta") + col("sinc_delta") + col("intel_delta") +
         col("fun_delta") + col("amb_delta") + col("shar_delta")) / 6,
        2   # round to 2 decimal places
    )
)

print("✓ Overall compatibility score computed")

# COMMAND ----------

# ── MUTUAL LIKE SCORE ──────────────────────────────────────────────────────
# like = how much person liked partner overall (1-10 scale)
# like_o = how much partner liked person overall (1-10 scale)
# mutual_like = average of both — high mutual like = strong match signal

speed_df = speed_df.withColumn(
    "mutual_like",
    round((col("like") + col("like_o")) / 2, 2)
)

print("✓ Mutual like score computed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering: Demographic Features
# MAGIC
# MAGIC Demographics can influence compatibility.
# MAGIC We engineer features that capture similarity rather than raw values.

# COMMAND ----------

# ── AGE GAP ────────────────────────────────────────────────────────────────
# Instead of raw ages, compute the gap between them
# A 2 year age gap is very different from a 20 year age gap
# abs() ensures the gap is always positive regardless of who is older

speed_df = speed_df.withColumn(
    "age_gap",
    abs(col("age") - col("age_o"))
)

print("✓ Age gap computed")

# COMMAND ----------

# ── SAME RACE FEATURE ──────────────────────────────────────────────────────
# samerace already exists in the dataset as 1 or 0
# 1 = same racial background, 0 = different
# We keep this as-is — it's already a clean binary feature

# No transformation needed — already clean from Silver
print("✓ samerace feature already clean")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering: Expectation vs Reality
# MAGIC
# MAGIC One of the most interesting features in this dataset:
# MAGIC prob = how likely does person THINK their partner will say yes?
# MAGIC prob_o = how likely does partner THINK person will say yes?
# MAGIC
# MAGIC If both people think the other will say yes AND they match,
# MAGIC that's a strong signal. We capture this as mutual confidence.

# COMMAND ----------

# ── MUTUAL CONFIDENCE SCORE ────────────────────────────────────────────────
# Average of both people's confidence that the other will say yes
# High mutual confidence = both people felt the connection

speed_df = speed_df.withColumn(
    "mutual_confidence",
    round((col("prob") + col("prob_o")) / 2, 2)
)

print("✓ Mutual confidence score computed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select Final Gold Features
# MAGIC
# MAGIC Now we select only the columns that go into ML training.
# MAGIC This is our final feature set — the exact columns XGBoost will learn from.
# MAGIC
# MAGIC **Target label:** match (1 = matched, 0 = did not match)
# MAGIC **Features:** everything else below

# COMMAND ----------

# ── SELECT ML FEATURES ─────────────────────────────────────────────────────
# These are the columns our XGBoost model will train on
# We drop the raw rating columns and keep only engineered features
# This prevents the model from "cheating" by memorizing raw IDs

gold_df = speed_df.select(
    # identifiers (not used in training but kept for reference)
    "iid",                  # person ID
    "pid",                  # partner ID

    # TARGET LABEL — what we are predicting
    "match",                # 1 = matched, 0 = did not match

    # demographic features
    "gender",               # 0 = female, 1 = male
    "age_gap",              # absolute age difference between pair
    "samerace",             # 1 if same race, 0 if not

    # mutual rating deltas (the core compatibility features)
    "attr_delta",           # attractiveness rating gap
    "sinc_delta",           # sincerity rating gap
    "intel_delta",          # intelligence rating gap
    "fun_delta",            # fun rating gap
    "amb_delta",            # ambition rating gap
    "shar_delta",           # shared interests rating gap

    # aggregate scores
    "compatibility_score",  # average of all 6 deltas
    "mutual_like",          # average of how much each liked the other
    "mutual_confidence",    # average of how likely each thought other would say yes
    "int_corr"              # pre-computed interest correlation from dataset
)

print(f"✓ Gold feature set: {len(gold_df.columns)} columns")
print(f"✓ Training rows: {gold_df.count()}")

# COMMAND ----------

# ── CHECK CLASS BALANCE ────────────────────────────────────────────────────
# Before writing we check how balanced our target label is
# If 95% of rows are match=0 and only 5% are match=1,
# the model will just predict 0 every time and be "95% accurate"
# This is called class imbalance and we need to know about it

match_counts = gold_df.groupBy("match").count()
match_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Gold Delta Table

# COMMAND ----------

# ── WRITE GOLD TABLE ───────────────────────────────────────────────────────
# This is the final ML-ready table
# XGBoost will read directly from this Delta Table during training

gold_df.write \
    .format("delta") \
    .mode("overwrite") \
    .save(f"{base_path}/gold/speed_dating_features")

print("✓ Gold feature table written successfully")

# COMMAND ----------

# ── VERIFY ─────────────────────────────────────────────────────────────────
gold_verify = spark.read.format("delta") \
    .load(f"{base_path}/gold/speed_dating_features")

print(f"✓ Gold table verified: {gold_verify.count()} rows, {len(gold_verify.columns)} columns")
gold_verify.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold Feature Engineering Complete
# MAGIC
# MAGIC Our Gold table is ready for ML training.
# MAGIC Every row is one speed dating encounter with engineered compatibility features.
# MAGIC Next step: 04_model_training.py
# MAGIC We will train an XGBoost classifier on this Gold table
# MAGIC to predict match probability between two people.