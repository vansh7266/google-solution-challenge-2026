import pandas as pd
import json
import os
import asyncio
import re
from .state import AuditState
from .ai_config import get_model, run_model_async

model = get_model("gemini-2.5-flash-lite")

async def agent_profiler(state: AuditState) -> AuditState:
    if state.get("demo_mode", False):
        await asyncio.sleep(0.2)
        return {
            **state,
            "domain": "criminal_justice",
            "domain_context": "In criminal justice, a Disparate Impact Ratio below 0.8 violates the EEOC 80% rule and may constitute unlawful discrimination under Title VII.",
            "sensitive_cols": ["race", "sex"],
        }

    df = pd.read_csv(state["dataset_path"])
    columns = df.columns.tolist()
    sample_vals = {col: df[col].dropna().unique()[:5].tolist() for col in columns}

    prompt = (
        f"Dataset columns: {columns}\n"
        f"Sample values: {sample_vals}\n"
        f"1. Identify sensitive columns (gender, race, age, religion, nationality, disability).\n"
        f"2. Detect domain: hiring | lending | healthcare | criminal_justice | education | other.\n"
        f"3. Write one sentence of legal/ethical context for that domain regarding bias.\n"
        f"Return ONLY valid JSON: "
        f'{{ "domain": "hiring", "sensitive_cols": ["age","gender"], "domain_context": "..." }}'
    )

    response = await run_model_async(model, prompt)
    detected_cols = []
    
    try:
        # Robust JSON extraction
        text = response.text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        raw = match.group(0) if match else text
        result = json.loads(raw)
        
        domain = result.get("domain", "unknown")
        context = result.get("domain_context", "")
        detected_cols = result.get("sensitive_cols", [])
        
    except Exception as e:
        domain = "unknown"
        context = f"Automatic detection fallback enabled. (Note: {str(e)[:40]}...)"
        detected_cols = []

    # CRITICAL FALLBACK: Keyword sweep to ensure we never miss sensitive attributes
    keywords = ["age", "gender", "sex", "race", "ethnic", "region", "relig", "nation", "disabil"]
    backup_cols = [c for c in columns if any(k in c.lower() for k in keywords)]
    
    # Merge AI detected and backup keywords, keeping unique values
    final_cols = list(set(detected_cols + backup_cols))
    
    # Prune non-existent columns just in case AI imagined some
    final_cols = [c for c in final_cols if c in columns]

    return {
        **state,
        "domain": domain,
        "domain_context": context,
        "sensitive_cols": final_cols,
    }