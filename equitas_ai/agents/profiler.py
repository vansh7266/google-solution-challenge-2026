# --- SECTION: Imports ---
import pandas as pd
import google.generativeai as genai
import json
import os
import asyncio
from .state import AuditState

# --- SECTION: Agent Initialization ---
# Ensure your GEMINI_API_KEY is in your .env file
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

# --- SECTION: Agent Logic ---
async def agent_profiler(state: AuditState) -> AuditState:
    df = pd.read_csv(state["dataset_path"])
    columns = df.columns.tolist()
    
    # Bypass API in demo mode for instant latency (Fix #2 from plan)
    if state.get("demo_mode", False):
        await asyncio.sleep(0.2)
        return {
            **state,
            "domain": "criminal_justice",
            "sensitive_cols": ["race", "sex"]
        }

    prompt = f"""
    Analyze these dataset columns: {columns}.
    1. Identify any sensitive columns (e.g., gender, race, age, religion, nationality).
    2. Guess the domain (hiring, lending, healthcare, criminal_justice, education).
    Return ONLY valid JSON: {{"domain": "hiring", "sensitive_cols": ["age", "gender"]}}
    """
    
    # Run Gemini asynchronously to avoid blocking the FastApi server
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, model.generate_content, prompt)
    
    try:
        raw_text = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw_text)
        
        return {
            **state,
            "domain": result.get("domain", "unknown"),
            "sensitive_cols": result.get("sensitive_cols", [])
        }
    except Exception:
        # Fallback if LLM parsing fails
        fallback_cols = [c for c in columns if c.lower() in ["age", "gender", "sex", "race", "ethnicity"]]
        return {
            **state,
            "domain": "unknown",
            "sensitive_cols": fallback_cols
        }