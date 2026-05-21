# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 - Model Training
# MAGIC
# MAGIC **What are we doing here?**
# MAGIC We take the Gold feature table we engineered in notebook 03
# MAGIC and train an XGBoost classification model on it.
# MAGIC
# MAGIC **What is XGBoost?**
# MAGIC XGBoost (Extreme Gradient Boosting) is one of the most powerful
# MAGIC and widely used ML algorithms for tabular data.
# MAGIC Think of it like building a team of 500 experts.
# MAGIC Each expert (decision tree) is weak on its own but learns from
# MAGIC the mistakes of the previous expert. Together they form a highly
# MAGIC accurate prediction machine.
# MAGIC
# MAGIC **What are we predicting?**
# MAGIC match = 1 (these two people will match)
# MAGIC match = 0 (these two people will not match)
# MAGIC Output: a probability between 0 and 1
# MAGIC Example: 0.73 = 73% chance these two people will match

# COMMAND ----------

# ── IMPORTS ────────────────────────────────────────────────────────────────
# We need both Spark (to read from Delta) and sklearn/xgboost (for ML)
# Spark reads the big distributed table, then we convert to pandas for training
# For our dataset size (~8k rows) pandas is fine after the Spark read

import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

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
# MAGIC ## Load Gold Table
# MAGIC
# MAGIC Read the ML-ready feature table from our Gold Delta layer.
# MAGIC We read it with Spark first (it's a Delta Table) then convert
# MAGIC to pandas for scikit-learn and XGBoost compatibility.
# MAGIC
# MAGIC Think of .toPandas() as downloading the distributed table
# MAGIC onto the driver node as a regular Python DataFrame.
# MAGIC Only do this when the data fits in memory — our 8k rows easily does.

# COMMAND ----------

# ── READ GOLD DELTA TABLE ──────────────────────────────────────────────────
gold_spark = spark.read.format("delta") \
    .load(f"{base_path}/gold/speed_dating_features")

print(f"Gold table loaded: {gold_spark.count()} rows")

# Convert to pandas for sklearn/xgboost
# .toPandas() pulls all data to the driver node as a regular DataFrame
df = gold_spark.toPandas()

print(f"Converted to pandas: {df.shape}")
print(f"\nClass distribution:")
print(df['match'].value_counts())
print(f"\nMatch rate: {df['match'].mean():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Features and Target
# MAGIC
# MAGIC In ML we separate our data into:
# MAGIC - X = features (the inputs the model learns from)
# MAGIC - y = target (what we want to predict)
# MAGIC
# MAGIC We also drop identifier columns (iid, pid) from features
# MAGIC because they are just ID numbers — the model shouldn't
# MAGIC memorize specific people, it should learn patterns.

# COMMAND ----------

# ── DEFINE FEATURE COLUMNS ─────────────────────────────────────────────────
# These are the exact columns we engineered in the Gold notebook
# Every column the model trains on must be numeric
# (XGBoost can't handle strings directly)

feature_cols = [
    "gender",               # 0 or 1
    "age_gap",              # float
    "samerace",             # 0 or 1
    "attr_delta",           # float
    "sinc_delta",           # float
    "intel_delta",          # float
    "fun_delta",            # float
    "amb_delta",            # float
    "shar_delta",           # float
    "compatibility_score",  # float
    "mutual_like",          # float
    "mutual_confidence",    # float
    "int_corr"              # float
]

# X = feature matrix (inputs)
X = df[feature_cols]

# y = target vector (what we predict)
# Our target is already 0 or 1 — perfect for binary classification
y = df['match']

print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# COMMAND ----------

# ── HANDLE REMAINING NULLS ─────────────────────────────────────────────────
# Even after Silver cleaning some nulls may remain
# XGBoost can actually handle NaN natively but let's be explicit
# fillna(0) for any remaining missing values

X = X.fillna(0)
print(f"Nulls after fill: {X.isnull().sum().sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Test Split
# MAGIC
# MAGIC We split our data into two sets:
# MAGIC - Training set (80%) — the model learns from this
# MAGIC - Test set (20%) — we evaluate the model on this
# MAGIC
# MAGIC The model NEVER sees the test set during training.
# MAGIC Think of it like studying (train) then taking an exam (test).
# MAGIC The exam questions are new — that's what makes it a fair evaluation.
# MAGIC
# MAGIC stratify=y ensures both sets have the same match/no-match ratio.
# MAGIC random_state=42 makes the split reproducible — same split every run.

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% held out for testing
    random_state=42,    # reproducible split
    stratify=y          # maintain class balance in both sets
)

print(f"Training set: {X_train.shape[0]} rows")
print(f"Test set: {X_test.shape[0]} rows")
print(f"Train match rate: {y_train.mean():.2%}")
print(f"Test match rate: {y_test.mean():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train XGBoost Model
# MAGIC
# MAGIC Key hyperparameters explained:
# MAGIC - n_estimators: number of trees (experts in our team)
# MAGIC - max_depth: how deep each tree can go (complexity per expert)
# MAGIC - learning_rate: how much each tree corrects the previous one
# MAGIC   (lower = more conservative = usually better)
# MAGIC - scale_pos_weight: handles class imbalance
# MAGIC   (if 80% are no-match, we weight matches higher so model pays attention)
# MAGIC - eval_metric: what metric to optimize during training
# MAGIC - use_label_encoder: suppress a deprecation warning

# COMMAND ----------

# ── COMPUTE CLASS WEIGHT ───────────────────────────────────────────────────
# If dataset has 80% no-match and 20% match,
# scale_pos_weight = 80/20 = 4
# This tells XGBoost "matches are 4x more important to get right"
# Without this the model would just predict 0 every time

neg_count = (y_train == 0).sum()  # number of no-match rows
pos_count = (y_train == 1).sum()  # number of match rows
scale_pos_weight = neg_count / pos_count

print(f"Class weight ratio: {scale_pos_weight:.2f}")

# COMMAND ----------

# ── TRAIN THE MODEL ────────────────────────────────────────────────────────
# MLflow automatically tracks this experiment
# Every parameter, metric, and the model itself gets logged
# You can view the results in the Databricks ML Experiments tab

with mlflow.start_run(run_name="matchiq-xgboost-v1"):

    # Define the model with our hyperparameters
    model = xgb.XGBClassifier(
        n_estimators=200,           # 200 trees
        max_depth=4,                # each tree max 4 levels deep
        learning_rate=0.1,          # conservative learning rate
        scale_pos_weight=scale_pos_weight,  # handle class imbalance
        eval_metric="auc",          # optimize for AUC during training
        random_state=42,            # reproducible results
        verbosity=0                 # suppress training output noise
    )

    # Train the model
    # fit() is where the actual learning happens
    # eval_set lets us monitor performance on test set during training
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # ── EVALUATE ───────────────────────────────────────────────────────────
    # Get predictions on the test set
    # predict() returns 0 or 1 (the class label)
    # predict_proba() returns [prob_0, prob_1] (the probabilities)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of match=1

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"MATCHIQ MODEL RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
    print(f"AUC Score: {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['No Match', 'Match']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ── LOG TO MLFLOW ──────────────────────────────────────────────────────
    # MLflow tracks everything so we can compare runs later
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 4)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("auc", auc)
    mlflow.xgboost.log_model(model, "matchiq-xgboost-model")

    print(f"\n✓ Model logged to MLflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance
# MAGIC
# MAGIC XGBoost tells us which features it found most useful.
# MAGIC This is one of the most valuable outputs — it tells us
# MAGIC WHAT actually predicts compatibility.
# MAGIC Higher score = more important for the prediction.

# COMMAND ----------

# ── FEATURE IMPORTANCE ─────────────────────────────────────────────────────
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFEATURE IMPORTANCE (what predicts compatibility):")
print("="*45)
for _, row in importance_df.iterrows():
    bar = "█" * int(row['importance'] * 100)
    print(f"{row['feature']:25s} {row['importance']:.4f} {bar}")

# COMMAND ----------

# ── SAVE MODEL LOCALLY ─────────────────────────────────────────────────────
# Save the trained model to DBFS so the agent can load it later
# DBFS = Databricks File System — shared storage across the cluster

import os
model_path = "/dbfs/matchiq/models/xgboost_model.json"
os.makedirs("/dbfs/matchiq/models", exist_ok=True)
model.save_model(model_path)

print(f"✓ Model saved to {model_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training Complete
# MAGIC
# MAGIC Our XGBoost classifier is trained and saved.
# MAGIC Next step: 05_model_evaluation.py
# MAGIC We will do deeper analysis of model performance
# MAGIC and test predictions on sample pairs.