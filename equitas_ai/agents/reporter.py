# --- SECTION: Imports ---
import os
import asyncio
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from textwrap import wrap
from pathlib import Path
from .state import AuditState

# --- SECTION: Agent Logic ---
async def agent_reporter(state: AuditState) -> AuditState:
    if state.get("demo_mode", False):
        await asyncio.sleep(0.4)
        return {**state, "report_path": "reports/demo_audit_report.pdf"}

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_filename = reports_dir / f"audit_report.pdf"

    # Run PDF generation in a background thread so it doesn't block FastAPI
    def create_pdf():
        c = canvas.Canvas(str(report_filename), pagesize=letter)
        
        # Headers
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, 750, "Equitas AI - Official Audit Report")
        
        # Data Profiling
        c.setFont("Helvetica", 12)
        c.drawString(50, 710, f"Domain Detected: {state.get('domain', 'N/A').upper()}")
        c.drawString(50, 690, f"Sensitive Attributes Flagged: {', '.join(state.get('sensitive_cols', []))}")
        
        # Metrics
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, 650, "Audit Results:")
        c.setFont("Helvetica", 12)
        c.drawString(50, 630, f"Initial Disparate Impact Score: {state.get('metrics', {}).get('Disparate Impact (DIR)', 'N/A')}")
        c.drawString(50, 610, f"Final Score Post-Remediation: {state.get('disparate_impact_score', 'N/A')}")
        
        # AI Explanation
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, 570, "Explainability Narrative:")
        c.setFont("Helvetica", 11)
        narrative = state.get("explanations", {}).get("narrative", "No narrative generated.")
        
        y_position = 550
        for line in wrap(narrative, width=80):
            c.drawString(50, y_position, line)
            y_position -= 20
            
        c.save()

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, create_pdf)

    return {
        **state,
        "report_path": str(report_filename)
    }