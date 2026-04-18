# --- SECTION: Imports ---
import os
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import AsyncGenerator

# LangGraph and State
from langgraph.graph import StateGraph, END
from agents.state import AuditState

# Agents
from agents.profiler import agent_profiler
from agents.detector import agent_bias_detector
from agents.explainer import agent_explainer
from agents.remediator import agent_remediator
from agents.reporter import agent_reporter


# --- SECTION: App Initialization ---
app = FastAPI(title="Equitas AI - Bias Detection Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# --- SECTION: LangGraph Pipeline Wiring ---
workflow = StateGraph(AuditState)

# 1. Add all agents as nodes
workflow.add_node("profiler", agent_profiler)
workflow.add_node("detector", agent_bias_detector)
workflow.add_node("explainer", agent_explainer)
workflow.add_node("remediator", agent_remediator)
workflow.add_node("reporter", agent_reporter)

# 2. Define the exact flow
workflow.set_entry_point("profiler")
workflow.add_edge("profiler", "detector")

# 3. The Decision Gate (Autonomous Routing)
def route_remediation(state: AuditState) -> str:
    di_score = state.get("disparate_impact_score", 1.0)
    iterations = state.get("iteration_count", 0)
    
    # If biased (score < 0.8) AND we haven't looped too many times, fix it
    if di_score < 0.8 and iterations < 2:
        return "remediator"
    # Otherwise, move to explanations and reporting
    return "explainer"

workflow.add_conditional_edges("detector", route_remediation)
workflow.add_edge("remediator", "detector") # Loop back to re-measure after fixing
workflow.add_edge("explainer", "reporter")
workflow.add_edge("reporter", END)

# Compile the agentic loop
equitas_engine = workflow.compile()


# --- SECTION: Endpoints ---
@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Saves dataset and returns the path for the audit pipeline."""
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename, "status": "uploaded", "path": str(file_path)}

# --- SECTION: SSE Streaming Engine ---
async def execute_audit_stream(filepath: str) -> AsyncGenerator[str, None]:
    """Runs LangGraph and streams real-time state changes to the UI."""
    
    initial_state = AuditState(
        dataset_path=filepath,
        domain="",
        sensitive_cols=[],
        metrics={},
        disparate_impact_score=1.0,
        explanations={},
        remediation_applied="None",
        iteration_count=0,
        demo_mode=DEMO_MODE
    )

    try:
        # astream() yields outputs as each node finishes
        async for output in equitas_engine.astream(initial_state):
            for node_name, state_update in output.items():
                
                # Format payload for the frontend UI ticker
                payload = {
                    "agent": node_name,
                    "status": "completed",
                    "current_score": state_update.get("disparate_impact_score"),
                    "iterations": state_update.get("iteration_count")
                }
                
                # Yield as an SSE string
                yield f"data: {json.dumps(payload)}\n\n"
                
        # Final payload to tell frontend to switch to the Dashboard view
        final_payload = {"agent": "pipeline", "status": "finished"}
        yield f"data: {json.dumps(final_payload)}\n\n"
        
    except Exception as e:
        error_payload = {"agent": "system", "status": "error", "message": str(e)}
        yield f"data: {json.dumps(error_payload)}\n\n"


@app.get("/audit/stream")
async def audit_stream(filepath: str):
    """The endpoint the frontend connects to for live updates."""
    return StreamingResponse(execute_audit_stream(filepath), media_type="text/event-stream")