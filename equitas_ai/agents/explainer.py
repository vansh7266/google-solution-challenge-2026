# --- SECTION: Imports ---
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai
import os
import asyncio
from .state import AuditState

# --- SECTION: Agent Initialization ---
model = genai.GenerativeModel('gemini-flash-lite-latest')

# --- SECTION: Agent Logic ---
async def agent_explainer(state: AuditState) -> AuditState:
    if state.get("demo_mode", False):
        await asyncio.sleep(0.8)
        return {
            **state, 
            "explanations": {
                "top_features": ["Age", "Income"],
                "narrative": "The model relies heavily on age. If this applicant were 10 years younger, their approval odds would increase by 24%."
            }
        }

    df = pd.read_csv(state["dataset_path"])
    
    # Layer 2 Latency Fix: Run SHAP on a max 500-row sample to prevent timeouts
    sample_df = df.sample(n=min(500, len(df)), random_state=42).dropna()
    target_col = sample_df.columns[-1]
    
    X = sample_df.drop(columns=[target_col])
    y = sample_df[target_col]

    # Fast label encoding for categorical variables so the Random Forest doesn't crash
    X_encoded = X.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)))
    y_encoded = LabelEncoder().fit_transform(y.astype(str))

    # Train a quick proxy model to evaluate feature importance
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_encoded, y_encoded)

    # --- SECTION: Foolproof Feature Importance ---
    try:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_encoded)
        
        # Handle SHAP version differences (list vs 3D array vs 2D array)
        if isinstance(shap_values, list):
            importances = abs(shap_values[-1]).mean(axis=0)
        elif len(shap_values.shape) == 3:
            importances = abs(shap_values[:, :, -1]).mean(axis=0)
        else:
            importances = abs(shap_values).mean(axis=0)
    except Exception:
        # Failsafe: Use native RF importances if SHAP array shapes break
        importances = rf.feature_importances_

    # Extract the top 3 most influential features
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values(by='importance', ascending=False)
    top_features = feature_importance['feature'].head(3).tolist()

    # Pass the math to Gemini to generate the human-readable counterfactual
    prompt = f"""
    You are an AI fairness auditor. We detected bias in a '{state.get('domain', 'general')}' dataset. 
    The top features driving the model's decisions are: {top_features}.
    The sensitive columns identified are: {state.get('sensitive_cols', [])}.
    Write a 2-sentence plain-English counterfactual explanation for a non-technical stakeholder explaining how the sensitive column influences the outcome.
    Do not use introductory phrases, just output the two sentences.
    """
    
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, model.generate_content, prompt)

    return {
        **state,
        "explanations": {
            "top_features": top_features,
            "narrative": response.text.strip()
        }
    }