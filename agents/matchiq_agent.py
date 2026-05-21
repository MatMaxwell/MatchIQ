# ── MATCHIQ LANGCHAIN AGENT ────────────────────────────────────────────────
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

# ── LOAD MODEL ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.json")

FEATURE_COLS = [
    "gender", "age_gap", "samerace",
    "attr_delta", "sinc_delta", "intel_delta",
    "fun_delta", "amb_delta", "shar_delta",
    "compatibility_score", "mutual_like",
    "mutual_confidence", "int_corr"
]

try:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    print("✓ XGBoost model loaded")
except Exception as e:
    print(f"⚠ Model not found: {e}")
    model = None

# ── TOOL 1: PREDICT COMPATIBILITY ─────────────────────────────────────────
@tool
def predict_compatibility(input_json: str) -> str:
    """
    Predicts compatibility probability between two people.
    Input: JSON string with age_gap, samerace, attr_delta, sinc_delta,
    intel_delta, fun_delta, amb_delta, shar_delta, mutual_like,
    mutual_confidence, int_corr, gender.
    Returns probability, prediction label, and explanation.
    """
    if model is None:
        return json.dumps({"error": "Model not loaded"})

    try:
        params = json.loads(input_json)
        delta_cols = ["attr_delta", "sinc_delta", "intel_delta",
                      "fun_delta", "amb_delta", "shar_delta"]
        compatibility_score = np.mean([params.get(c, 0) for c in delta_cols])

        features = pd.DataFrame([{
            "gender": params.get("gender", 1),
            "age_gap": params.get("age_gap", 0),
            "samerace": params.get("samerace", 0),
            "attr_delta": params.get("attr_delta", 0),
            "sinc_delta": params.get("sinc_delta", 0),
            "intel_delta": params.get("intel_delta", 0),
            "fun_delta": params.get("fun_delta", 0),
            "amb_delta": params.get("amb_delta", 0),
            "shar_delta": params.get("shar_delta", 0),
            "compatibility_score": compatibility_score,
            "mutual_like": params.get("mutual_like", 5.0),
            "mutual_confidence": params.get("mutual_confidence", 5.0),
            "int_corr": params.get("int_corr", 0.0)
        }])

        prob = model.predict_proba(features)[0][1]
        prediction = int(prob >= 0.5)

        signals = []
        if params.get("age_gap", 0) <= 3:
            signals.append("close in age")
        elif params.get("age_gap", 0) >= 10:
            signals.append("significant age gap")
        if params.get("samerace", 0) == 1:
            signals.append("shared racial background")
        if compatibility_score <= 2:
            signals.append("very aligned mutual ratings")
        elif compatibility_score >= 5:
            signals.append("large gaps in mutual ratings")
        if params.get("mutual_like", 5) >= 7:
            signals.append("strong mutual liking")
        elif params.get("mutual_like", 5) <= 3:
            signals.append("low mutual interest")

        explanation = f"Key signals: {', '.join(signals)}." if signals else "No strong signals detected."

        return json.dumps({
            "probability": round(float(prob), 4),
            "percentage": f"{prob:.1%}",
            "label": "MATCH" if prediction == 1 else "NO MATCH",
            "explanation": explanation
        })

    except Exception as e:
        return json.dumps({"error": str(e)})

# ── TOOL 2: FEATURE IMPORTANCE ─────────────────────────────────────────────
@tool
def get_feature_importance(dummy: str = "") -> str:
    """
    Returns ranked feature importance from the trained model.
    Use when asked what factors matter most for compatibility.
    """
    if model is None:
        return "Model not loaded."

    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    descriptions = {
        "mutual_like": "how much each person liked the other overall",
        "mutual_confidence": "confidence each felt the other would say yes",
        "compatibility_score": "average gap across all six rating attributes",
        "attr_delta": "gap in attractiveness ratings",
        "sinc_delta": "gap in sincerity ratings",
        "intel_delta": "gap in intelligence ratings",
        "fun_delta": "gap in fun ratings",
        "amb_delta": "gap in ambition ratings",
        "shar_delta": "gap in shared interests ratings",
        "age_gap": "absolute age difference",
        "samerace": "shared racial background",
        "int_corr": "correlation of stated interests",
        "gender": "gender of person 1"
    }

    result = "Feature importance (what predicts compatibility):\n\n"
    for i, (feat, imp) in enumerate(sorted_imp, 1):
        desc = descriptions.get(feat, feat)
        result += f"{i}. {desc}: {imp:.4f}\n"
    return result

# ── BUILD AGENT ────────────────────────────────────────────────────────────
def build_agent():
    llm = ChatAnthropic(
        model="claude-haiku-4-5",
        temperature=0,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    system_prompt = """You are MatchIQ, an AI compatibility analyst powered by a machine learning model 
trained on real speed dating data from Columbia University.

When given information about two people, use the predict_compatibility tool to calculate 
their match probability, then explain the result warmly and conversationally.

When asked what factors matter most, use the get_feature_importance tool.

Always be encouraging — compatibility is complex and the model captures patterns, 
not certainties. Human connection has dimensions no model can fully capture."""

    return create_react_agent(llm, [predict_compatibility, get_feature_importance], prompt=system_prompt)

# ── TEST ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nStarting MatchIQ Agent...")
    agent = build_agent()

    print("\n" + "="*60)
    print("TEST: Compatibility prediction")
    print("="*60)

    response = agent.invoke({
        "messages": [{
            "role": "user",
            "content": """Can you assess compatibility between these two people?
            Age gap: 2 years, same race: yes,
            attractiveness gap: 1.5, sincerity gap: 0.5,
            intelligence gap: 1.0, fun gap: 0.5,
            ambition gap: 2.0, shared interests gap: 1.0,
            mutual liking: 8.5/10, mutual confidence: 7.5/10,
            interest correlation: 0.6"""
        }]
    })

    print(f"\nAgent Response:\n{response['messages'][-1].content}")