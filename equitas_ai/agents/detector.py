# --- SECTION: Imports ---
import pandas as pd
import asyncio
from .state import AuditState

# --- SECTION: Agent Logic ---
async def agent_bias_detector(state: AuditState) -> AuditState:
    # Bypass heavy math in demo mode for instant 1-second results
    if state.get("demo_mode", False):
        await asyncio.sleep(0.5)
        return {
            **state, 
            "disparate_impact_score": 0.64, 
            "metrics": {"Disparate Impact (DIR)": 0.64, "Equal Opportunity": 0.71}
        }

    df = pd.read_csv(state["dataset_path"])
    sensitive_cols = state.get("sensitive_cols", [])
    
    if not sensitive_cols:
        return {**state, "disparate_impact_score": 1.0, "metrics": {}}

    # Heuristic: Assume the last column in the dataset is the target outcome (standard for CSVs)
    target_col = df.columns[-1]
    
    # Evaluate the primary sensitive feature found by Agent 1
    sensitive_col = sensitive_cols[0] 

    # Calculate Disparate Impact Ratio (DIR)
    # DIR = (Success rate of unprivileged group) / (Success rate of privileged group)
    rates = df.groupby(sensitive_col)[target_col].mean().to_dict()
    
    if len(rates) >= 2:
        max_rate = max(rates.values())
        min_rate = min(rates.values())
        # Prevent division by zero
        di_score = min_rate / max_rate if max_rate > 0 else 1.0 
    else:
        di_score = 1.0

    metrics = {
        "Disparate Impact (DIR)": round(di_score, 3),
        "Analyzed Feature": sensitive_col
    }

    return {
        **state,
        "disparate_impact_score": round(di_score, 3),
        "metrics": metrics
    }