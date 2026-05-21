# MatchIQ

I built MatchIQ to get hands-on experience with a real end-to-end data + AI pipeline. 
The idea is simple — take two messy real-world dating datasets, run them through a full 
data engineering process, and use the cleaned data to predict whether two people will 
be compatible before they ever meet.

## What it does
Raw data from the Columbia Speed Dating Experiment and OKCupid gets ingested and pushed 
through a Bronze → Silver → Gold medallion pipeline on Azure Databricks using PySpark. 
Each layer progressively cleans, transforms, and engineers features until the data lands 
in a Delta Table data warehouse that the AI side can consume.

From there an XGBoost classification model predicts compatibility probability between 
two people. That model gets wrapped in a LangChain agent powered by Claude that can 
explain predictions in plain english — not just a number but actual reasoning.

The Streamlit app ties it all together with analytics dashboards showing patterns in 
the transformed data plus the chatbot interface where you can input two profiles and 
see their compatibility probability before they ever meet.

## Stack
- Azure Databricks + PySpark (medallion pipeline)
- Azure Data Lake Storage Gen2 (data lake)
- Delta Tables (data warehouse)
- Azure ML (model training + deployment)
- XGBoost + Scikit-learn (compatibility classifier)
- LangChain + Claude (AI agent)
- Streamlit (analytics dashboard + chatbot UI)

## Datasets
- Columbia Speed Dating Experiment — 8,378 rows, 195 columns, real match outcomes
- OKCupid Profiles — 68,371 rows, rich personality and lifestyle features

## Project Structure
- `data/` — raw, bronze, silver, gold layers
- `pipelines/` — PySpark transformation scripts
- `notebooks/` — Databricks notebooks
- `agents/` — LangChain agent code
- `models/` — Azure ML training scripts
- `streamlit/` — analytics dashboard and chatbot UI
- `docs/` — architecture diagrams
