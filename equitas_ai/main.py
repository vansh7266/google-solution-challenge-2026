import os
import json
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import AsyncGenerator, Any
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import StateGraph, END

from agents.state import AuditState
from agents.profiler import agent_profiler
from agents.detector import agent_bias_detector
from agents.explainer import agent_explainer
from agents.remediator import agent_remediator
from agents.reporter import agent_reporter


class SafeEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if hasattr(obj, "item"): return obj.item()
        return str(obj)


def sjson(data: Any) -> str:
    return json.dumps(data, cls=SafeEncoder)


app = FastAPI(title="Equitas AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEMO_MODE_ENV = os.getenv("DEMO_MODE", "false").lower() == "true"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


workflow = StateGraph(AuditState)
workflow.add_node("profiler", agent_profiler)
workflow.add_node("detector", agent_bias_detector)
workflow.add_node("explainer", agent_explainer)
workflow.add_node("remediator", agent_remediator)
workflow.add_node("reporter", agent_reporter)

workflow.set_entry_point("profiler")
workflow.add_edge("profiler", "detector")


def route_after_detector(state: AuditState) -> str:
    if state.get("disparate_impact_score", 1.0) < 0.8 and state.get("iteration_count", 0) < 2:
        return "remediator"
    return "explainer"


workflow.add_conditional_edges("detector", route_after_detector)
workflow.add_edge("remediator", "detector")
workflow.add_edge("explainer", "reporter")
workflow.add_edge("reporter", END)

equitas_engine = workflow.compile()


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    preview = {}
    try:
        df = pd.read_csv(file_path)
        preview = {
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "head": df.head(5).fillna("N/A").astype(str).to_dict(orient="records"),
            "missing": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        }
    except Exception as e:
        preview = {"error": str(e)}

    return {"filename": file.filename, "status": "uploaded", "path": str(file_path), "preview": preview}


async def run_audit_stream(filepath: str, demo: bool) -> AsyncGenerator[str, None]:
    initial: AuditState = {
        "dataset_path": filepath,
        "demo_mode": demo,
        "domain": "",
        "domain_context": "",
        "sensitive_cols": [],
        "metrics": {},
        "initial_metrics": {},
        "disparate_impact_score": 1.0,
        "initial_disparate_impact": 1.0,
        "intersectional_matrix": {},
        "shap_features": [],
        "explanations": {},
        "remediation_applied": "None",
        "iteration_count": 0,
        "report_path": "",
    }

    accumulated = dict(initial)

    try:
        async for output in equitas_engine.astream(initial):
            for node_name, state_update in output.items():
                accumulated.update(state_update)
                payload = {
                    "agent": node_name,
                    "status": "completed",
                    "current_score": accumulated.get("disparate_impact_score"),
                    "iterations": accumulated.get("iteration_count", 0),
                }
                yield f"data: {sjson(payload)}\n\n"

        final = {"agent": "pipeline", "status": "finished", "result": accumulated}
        yield f"data: {sjson(final)}\n\n"

    except Exception as e:
        yield f"data: {sjson({'agent': 'system', 'status': 'error', 'message': str(e)})}\n\n"


@app.get("/audit/stream")
async def audit_stream(filepath: str = Query(...), demo: bool = Query(False)):
    return StreamingResponse(
        run_audit_stream(filepath, demo or DEMO_MODE_ENV),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/history")
async def get_history():
    history_path = Path("audit_history")
    history_path.mkdir(exist_ok=True)
    audits = []
    for f in sorted(history_path.glob("audit_*.json"), reverse=True)[:20]:
        try:
            with open(f) as fh:
                data = json.load(fh)
                data["timestamp"] = f.stem.replace("audit_", "")
                audits.append(data)
        except Exception:
            continue
    return {"audits": audits}


@app.get("/download-report")
async def download_report():
    report_path = Path("reports/audit_report.pdf")
    if report_path.exists():
        return FileResponse(
            path=report_path,
            filename="Equitas_AI_Report.pdf",
            media_type="application/pdf",
        )
    return {"error": "No report found. Run an audit first."}

@app.get("/sample/{name}")
async def load_sample(name: str):
    samples = {
        "compas":         "sample_data/compas.csv",
        "adult_income":   "sample_data/adult_income.csv",
        "german_credit":  "sample_data/german_credit.csv",
    }
    if name not in samples:
        return {"error": "Unknown sample"}
    path = Path(samples[name])
    if not path.exists():
        return {"error": "Sample file not found"}
    preview = {}
    try:
        df = pd.read_csv(path)
        preview = {
            "columns": df.columns.tolist(),
            "dtypes":  df.dtypes.astype(str).to_dict(),
            "shape":   [int(df.shape[0]), int(df.shape[1])],
            "head":    df.head(5).fillna("N/A").astype(str).to_dict(orient="records"),
            "missing": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        }
    except Exception as e:
        preview = {"error": str(e)}
    return {"filename": path.name, "status": "loaded", "path": str(path), "preview": preview}


@app.post("/ask")
async def ask_gemini(payload: dict):
    question = payload.get("question", "")
    context  = payload.get("context", {})
    if not question:
        return {"answer": "No question provided."}

    ctx_str = json.dumps(context, cls=SafeEncoder)
    prompt = (
        f"You are an AI fairness auditor. Here is the completed bias audit result:\n{ctx_str}\n\n"
        f"Answer this question from the user in 2-3 clear sentences using the audit data above:\n{question}"
    )
    try:
        import google.generativeai as genai
        m = genai.GenerativeModel("gemini-2.0-flash")
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, m.generate_content, prompt)
        return {"answer": response.text.strip()}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}

@app.post("/model-card")
async def generate_model_card(payload: dict):
    result = payload.get("result", {})
    if not result:
        return {"error": "No audit result provided."}

    domain        = result.get("domain", "unknown")
    sensitive     = ", ".join(result.get("sensitive_cols", []))
    di            = result.get("disparate_impact_score", 0)
    metrics       = result.get("metrics", {})
    narrative     = result.get("explanations", {}).get("narrative", "")
    fix           = result.get("remediation_applied", "None")
    metrics_str   = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])

    prompt = (
        f"Generate a structured Model Card for an AI system audited by Equitas AI.\n"
        f"Domain: {domain}\n"
        f"Sensitive attributes: {sensitive}\n"
        f"Disparate Impact Ratio: {di}\n"
        f"Fairness metrics:\n{metrics_str}\n"
        f"Explainability: {narrative}\n"
        f"Remediation applied: {fix}\n\n"
        f"Write the Model Card with these exact sections:\n"
        f"## Model Details\n## Intended Use\n## Factors\n## Metrics\n## Evaluation Data\n"
        f"## Ethical Considerations\n## Recommendations\n\n"
        f"Be concise. Each section 2-3 sentences. Plain English."
    )

    try:
        import google.generativeai as genai
        m        = genai.GenerativeModel("gemini-2.0-flash")
        loop     = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, m.generate_content, prompt)
        return {"card": response.text.strip()}
    except Exception as e:
        return {"error": str(e)}