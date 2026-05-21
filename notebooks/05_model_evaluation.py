# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 05 - Model Evaluation
# MAGIC
# MAGIC **What are we doing here?**
# MAGIC We load our trained XGBoost model and test it on real pairs.
# MAGIC We also build a prediction function that the LangChain agent
# MAGIC will call when a user asks "will these two people match?"
# MAGIC
# MAGIC Think of this notebook as the "quality control" step —
# MAGIC before we ship the model to production we stress test it
# MAGIC and make sure it makes sensible predictions.

# COMMAND ----------

# ── IMPORTS ────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import xgboost as xgb
import json

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

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
# MAGIC ## Load Trained Model

# COMMAND ----------

# ── LOAD MODEL FROM DBFS ───────────────────────────────────────────────────
# We saved the model in notebook 04 — now we load it back
# This is how the agent will load it too

model = xgb.XGBClassifier()
model.load_model("/dbfs/matchiq/models/xgboost_model.json")
print("✓ Model loaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Gold Table for Evaluation

# COMMAND ----------

# ── READ GOLD TABLE ────────────────────────────────────────────────────────
gold_spark = spark.read.format("delta") \
    .load(f"{base_path}/gold/speed_dating_features")

df = gold_spark.toPandas()

feature_cols = [
    "gender", "age_gap", "samerace",
    "attr_delta", "sinc_delta", "intel_delta",
    "fun_delta", "amb_delta", "shar_delta",
    "compatibility_score", "mutual_like",
    "mutual_confidence", "int_corr"
]

X = df[feature_cols].fillna(0)
y = df['match']

print(f"Evaluation data: {X.shape[0]} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Full Dataset Evaluation
# MAGIC
# MAGIC We evaluate the model across multiple metrics:
# MAGIC
# MAGIC - **Accuracy** — what % of predictions were correct overall
# MAGIC - **AUC** — how well the model separates matches from non-matches
# MAGIC   (1.0 = perfect, 0.5 = random guessing)
# MAGIC - **Precision** — of all predicted matches, how many were real matches?
# MAGIC - **Recall** — of all real matches, how many did we catch?
# MAGIC - **F1** — balance between precision and recall

# COMMAND ----------

# ── COMPUTE ALL METRICS ────────────────────────────────────────────────────
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

accuracy  = accuracy_score(y, y_pred)
auc       = roc_auc_score(y, y_prob)
precision = precision_score(y, y_pred)
recall    = recall_score(y, y_pred)
f1        = f1_score(y, y_pred)

print(f"\n{'='*50}")
print(f"MATCHIQ FULL EVALUATION REPORT")
print(f"{'='*50}")
print(f"Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
print(f"AUC:       {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test on Sample Pairs
# MAGIC
# MAGIC Let's manually test the model on a few real pairs from the dataset
# MAGIC and see if the predictions make intuitive sense.
# MAGIC This is a sanity check — if the model predicts 95% match probability
# MAGIC for a pair where both rated each other 1/10, something is wrong.

# COMMAND ----------

# ── SAMPLE MATCHED PAIRS ───────────────────────────────────────────────────
# Get 5 pairs that actually matched and see what the model predicts
matched_pairs = df[df['match'] == 1].head(5)
matched_X = matched_pairs[feature_cols].fillna(0)
matched_probs = model.predict_proba(matched_X)[:, 1]

print("ACTUAL MATCHES — model predictions:")
print("="*45)
for i, (_, row) in enumerate(matched_pairs.iterrows()):
    prob = matched_probs[i]
    print(f"Pair {i+1}: {prob:.2%} match probability (actual: MATCH ✓)")

# COMMAND ----------

# ── SAMPLE NON-MATCHED PAIRS ───────────────────────────────────────────────
# Get 5 pairs that did NOT match
non_matched_pairs = df[df['match'] == 0].head(5)
non_matched_X = non_matched_pairs[feature_cols].fillna(0)
non_matched_probs = model.predict_proba(non_matched_X)[:, 1]

print("\nACTUAL NON-MATCHES — model predictions:")
print("="*45)
for i, (_, row) in enumerate(non_matched_pairs.iterrows()):
    prob = non_matched_probs[i]
    print(f"Pair {i+1}: {prob:.2%} match probability (actual: NO MATCH ✗)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Prediction Function
# MAGIC
# MAGIC This is the function the LangChain agent will call.
# MAGIC It takes two people's features as input and returns
# MAGIC a match probability plus a plain english explanation.
# MAGIC
# MAGIC Think of it as the bridge between the ML model and the chatbot.

# COMMAND ----------

def predict_compatibility(
    age_gap,
    samerace,
    attr_delta,
    sinc_delta,
    intel_delta,
    fun_delta,
    amb_delta,
    shar_delta,
    mutual_like=5.0,
    mutual_confidence=5.0,
    int_corr=0.0,
    gender=1
):
    """
    Predict compatibility probability between two people.

    Parameters:
    -----------
    age_gap : float
        Absolute age difference between the two people
    samerace : int
        1 if same racial background, 0 if different
    attr_delta : float
        Gap in attractiveness ratings (0-10 scale)
    sinc_delta : float
        Gap in sincerity ratings (0-10 scale)
    intel_delta : float
        Gap in intelligence ratings (0-10 scale)
    fun_delta : float
        Gap in fun ratings (0-10 scale)
    amb_delta : float
        Gap in ambition ratings (0-10 scale)
    shar_delta : float
        Gap in shared interests ratings (0-10 scale)
    mutual_like : float
        Average of how much each person liked the other (1-10)
    mutual_confidence : float
        Average confidence both felt the other would say yes (1-10)
    int_corr : float
        Correlation of interests (-1 to 1)
    gender : int
        Gender of person 1 (0=female, 1=male)

    Returns:
    --------
    dict with probability, prediction, and explanation
    """

    # Build feature vector in exact same order as training
    features = pd.DataFrame([{
        "gender": gender,
        "age_gap": age_gap,
        "samerace": samerace,
        "attr_delta": attr_delta,
        "sinc_delta": sinc_delta,
        "intel_delta": intel_delta,
        "fun_delta": fun_delta,
        "amb_delta": amb_delta,
        "shar_delta": shar_delta,
        "compatibility_score": np.mean([
            attr_delta, sinc_delta, intel_delta,
            fun_delta, amb_delta, shar_delta
        ]),
        "mutual_like": mutual_like,
        "mutual_confidence": mutual_confidence,
        "int_corr": int_corr
    }])

    # Get probability from model
    prob = model.predict_proba(features)[0][1]
    prediction = int(prob >= 0.5)

    # Build plain english explanation
    # This is what the LangChain agent will use to explain the prediction
    signals = []

    if age_gap <= 3:
        signals.append("close in age")
    elif age_gap >= 10:
        signals.append("significant age gap")

    if samerace == 1:
        signals.append("shared racial background")

    avg_delta = np.mean([attr_delta, sinc_delta, intel_delta,
                         fun_delta, amb_delta, shar_delta])
    if avg_delta <= 2:
        signals.append("very similar mutual ratings")
    elif avg_delta >= 5:
        signals.append("large gaps in how they rated each other")

    if mutual_like >= 7:
        signals.append("strong mutual liking")
    elif mutual_like <= 3:
        signals.append("low mutual interest")

    if mutual_confidence >= 7:
        signals.append("both felt confident the other would say yes")

    explanation = f"Match probability: {prob:.2%}. " + \
                  f"Key signals: {', '.join(signals)}." if signals else \
                  f"Match probability: {prob:.2%}."

    return {
        "probability": round(float(prob), 4),
        "prediction": prediction,
        "label": "MATCH" if prediction == 1 else "NO MATCH",
        "explanation": explanation
    }

# COMMAND ----------

# ── TEST THE PREDICTION FUNCTION ───────────────────────────────────────────
# Test case 1: Strong compatibility signals
result1 = predict_compatibility(
    age_gap=2,          # close in age
    samerace=1,         # same background
    attr_delta=1.0,     # similar attractiveness ratings
    sinc_delta=0.5,     # very similar sincerity
    intel_delta=1.0,    # similar intelligence ratings
    fun_delta=0.5,      # similar fun ratings
    amb_delta=1.0,      # similar ambition
    shar_delta=0.5,     # very similar interests
    mutual_like=8.5,    # both liked each other a lot
    mutual_confidence=8.0,  # both felt confident
    int_corr=0.6        # strong interest correlation
)

print("TEST 1 — Strong compatibility:")
print(json.dumps(result1, indent=2))

# COMMAND ----------

# Test case 2: Weak compatibility signals
result2 = predict_compatibility(
    age_gap=15,         # big age gap
    samerace=0,         # different backgrounds
    attr_delta=6.0,     # very different attractiveness ratings
    sinc_delta=4.0,     # different sincerity ratings
    intel_delta=5.0,    # different intelligence ratings
    fun_delta=6.0,      # very different fun ratings
    amb_delta=3.0,      # different ambition
    shar_delta=5.0,     # different interests
    mutual_like=2.5,    # neither liked the other much
    mutual_confidence=2.0,  # neither felt confident
    int_corr=-0.3       # negative interest correlation
)

print("\nTEST 2 — Weak compatibility:")
print(json.dumps(result2, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Prediction Function Config
# MAGIC
# MAGIC Save the feature column order and model path to a config file
# MAGIC so the LangChain agent can load everything consistently.

# COMMAND ----------

config = {
    "model_path": "/dbfs/matchiq/models/xgboost_model.json",
    "feature_cols": feature_cols,
    "version": "1.0.0"
}

config_path = "/dbfs/matchiq/models/model_config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Model config saved to {config_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Complete
# MAGIC
# MAGIC Our model is evaluated and the prediction function is ready.
# MAGIC Next step: Build the LangChain agent in agents/matchiq_agent.py
# MAGIC The agent will use predict_compatibility() as one of its tools
# MAGIC and explain predictions in plain english using Claude.