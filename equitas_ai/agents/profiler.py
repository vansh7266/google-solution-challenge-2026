import pandas as pd
import google.generativeai as genai
import json
import os
import asyncio
from .state import AuditState

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash-lite")


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

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, model.generate_content, prompt)

    try:
        raw = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        return {
            **state,
            "domain": result.get("domain", "unknown"),
            "domain_context": result.get("domain_context", ""),
            "sensitive_cols": result.get("sensitive_cols", []),
        }
    except Exception:
        fallback = [c for c in columns if c.lower() in ["age", "gender", "sex", "race", "ethnicity", "religion", "nationality"]]
        return {
            **state,
            "domain": "unknown",
            "domain_context": "",
            "sensitive_cols": fallback,
        }