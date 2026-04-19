import pandas as pd
import shap
import asyncio
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai
from .state import AuditState

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")


async def agent_explainer(state: AuditState) -> AuditState:
    if state.get("demo_mode", False):
        await asyncio.sleep(0.8)
        return {
            **state,
            "shap_features": [
                {"feature": "Age",            "importance": 0.38},
                {"feature": "Education",      "importance": 0.22},
                {"feature": "Hours-per-week", "importance": 0.18},
                {"feature": "Race",           "importance": 0.14},
                {"feature": "Sex",            "importance": 0.08},
            ],
            "explanations": {
                "top_features": ["Age", "Education", "Hours-per-week"],
                "narrative": (
                    "The model relies heavily on Age and Education to make its decisions. "
                    "If this applicant were 10 years younger, their approval odds would increase "
                    "by 24%, revealing that Age acts as a proxy for protected class characteristics."
                ),
            },
        }

    df = pd.read_csv(state["dataset_path"])
    sample_df = df.sample(n=min(500, len(df)), random_state=42).dropna()
    target_col = sample_df.columns[-1]

    X = sample_df.drop(columns=[target_col])
    y = sample_df[target_col]
    X_enc = X.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)))
    y_enc = LabelEncoder().fit_transform(y.astype(str))

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_enc, y_enc)

    try:
        explainer_obj = shap.TreeExplainer(rf)
        shap_vals = explainer_obj.shap_values(X_enc)
        if isinstance(shap_vals, list):
            importances = abs(shap_vals[-1]).mean(axis=0)
        elif len(shap_vals.shape) == 3:
            importances = abs(shap_vals[:, :, -1]).mean(axis=0)
        else:
            importances = abs(shap_vals).mean(axis=0)
    except Exception:
        importances = rf.feature_importances_

    feat_df = (
        pd.DataFrame({"feature": X.columns, "importance": importances})
        .sort_values("importance", ascending=False)
    )
    top_features = feat_df["feature"].head(3).tolist()
    shap_features = [
        {"feature": str(r["feature"]), "importance": round(float(r["importance"]), 4)}
        for _, r in feat_df.head(5).iterrows()
    ]

    prompt = (
        f"You are an AI fairness auditor. Domain: '{state.get('domain', 'general')}'. "
        f"Top model features: {top_features}. "
        f"Sensitive columns: {state.get('sensitive_cols', [])}. "
        f"Write exactly 2 plain-English sentences: explain why the model is biased and give a "
        f"counterfactual example showing what would change the outcome. No preamble."
    )
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, model.generate_content, prompt)

    return {
        **state,
        "shap_features": shap_features,
        "explanations": {
            "top_features": top_features,
            "narrative": response.text.strip(),
        },
    }