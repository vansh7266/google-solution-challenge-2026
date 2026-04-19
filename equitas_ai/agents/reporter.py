import asyncio
from pathlib import Path
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors as rlc
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from .state import AuditState

G_BLUE   = "#4285F4"
G_RED    = "#EA4335"
G_GREEN  = "#34A853"
G_YELLOW = "#FBBC04"
BG_DARK  = "#060B18"
BG_CARD  = "#0D1626"
BG_CARD2 = "#111D35"
TXT      = "#E8EAED"
MUTED    = "#9AA0A6"

RL_BLUE   = rlc.HexColor(G_BLUE)
RL_RED    = rlc.HexColor(G_RED)
RL_GREEN  = rlc.HexColor(G_GREEN)
RL_YELLOW = rlc.HexColor(G_YELLOW)
RL_DARK   = rlc.HexColor(BG_DARK)
RL_CARD   = rlc.HexColor(BG_CARD)
RL_CARD2  = rlc.HexColor(BG_CARD2)
RL_BORDER = rlc.HexColor("#1E3A5F")
RL_TEXT   = rlc.HexColor(TXT)
RL_MUTED  = rlc.HexColor(MUTED)


def _score_color_rl(score: float):
    if score < 0.6:
        return RL_RED
    if score < 0.8:
        return RL_YELLOW
    return RL_GREEN


def _score_color_mpl(score: float, is_dir: bool = True) -> str:
    if is_dir:
        return G_RED if score < 0.6 else (G_YELLOW if score < 0.8 else G_GREEN)
    return G_RED if score > 0.2 else (G_YELLOW if score > 0.1 else G_GREEN)


def _buf_gauge(before: float, after: float) -> BytesIO:
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.6))
    fig.patch.set_facecolor(BG_DARK)
    for ax, score, label in zip(axes, [before, after], ["Before", "After"]):
        ax.set_facecolor(BG_DARK)
        color = _score_color_mpl(score)
        theta_bg = np.linspace(np.pi, 0, 200)
        ax.plot(np.cos(theta_bg), np.sin(theta_bg), color="#1A2B4A", linewidth=14, solid_capstyle="round")
        end_angle = np.pi - np.pi * min(max(score, 0), 1)
        theta = np.linspace(np.pi, end_angle, 200)
        ax.plot(np.cos(theta), np.sin(theta), color=color, linewidth=14, solid_capstyle="round", zorder=2)
        ax.text(0, -0.05, f"{score:.2f}", ha="center", va="center", fontsize=22, fontweight="bold", color=TXT)
        ax.text(0, -0.55, label, ha="center", va="center", fontsize=10, color=MUTED)
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-0.75, 1.2)
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor=BG_DARK, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _buf_metrics(metrics: dict) -> BytesIO:
    valid = {k: float(v) for k, v in metrics.items() if v is not None}
    if not valid:
        return None
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor(BG_CARD)
    ax.set_facecolor(BG_CARD2)
    labels = list(valid.keys())
    values = list(valid.values())
    colors = [_score_color_mpl(v, is_dir=("DIR" in k)) for k, v in valid.items()]
    bars = ax.barh(labels, values, color=colors, height=0.5, edgecolor="none")
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                va="center", color=TXT, fontsize=8, fontweight="bold")
    ax.axvline(0.8, color=G_BLUE, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(0.81, len(labels) - 0.5, "Threshold 0.8", color=G_BLUE, fontsize=7, va="top")
    ax.set_xlim(0, 1.2)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(pad=0.8)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor=BG_CARD, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _buf_shap(shap_features: list) -> BytesIO:
    if not shap_features:
        return None
    fig, ax = plt.subplots(figsize=(6, 2.4))
    fig.patch.set_facecolor(BG_CARD)
    ax.set_facecolor(BG_CARD2)
    feats = [f["feature"] for f in shap_features[:5]]
    vals = [f["importance"] for f in shap_features[:5]]
    ax.barh(feats, vals, color=G_BLUE, height=0.5, edgecolor="none", alpha=0.9)
    ax.tick_params(colors=TXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(pad=0.8)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor=BG_CARD, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _buf_comparison(init_m: dict, final_m: dict) -> BytesIO:
    keys = list(init_m.keys())
    if not keys:
        return None
    x = np.arange(len(keys))
    before_vals = [float(init_m.get(k, 0)) for k in keys]
    after_vals = [float(final_m.get(k, 0)) for k in keys]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    fig.patch.set_facecolor(BG_CARD)
    ax.set_facecolor(BG_CARD2)
    w = 0.35
    ax.bar(x - w / 2, before_vals, width=w, color=G_RED, alpha=0.85, label="Before", edgecolor="none")
    ax.bar(x + w / 2, after_vals, width=w, color=G_GREEN, alpha=0.85, label="After", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace(" ", "\n") for k in keys], fontsize=7, color=TXT)
    ax.tick_params(colors=TXT, labelsize=8)
    ax.legend(fontsize=8, labelcolor=TXT, facecolor=BG_CARD2, edgecolor="none")
    ax.axhline(0.8, color=G_BLUE, linestyle="--", linewidth=0.7, alpha=0.6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(pad=0.8)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor=BG_CARD, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _styles():
    return {
        "title":   ParagraphStyle("T", fontName="Helvetica-Bold", fontSize=24, textColor=RL_TEXT, alignment=TA_CENTER, spaceAfter=4),
        "sub":     ParagraphStyle("S", fontName="Helvetica", fontSize=11, textColor=RL_MUTED, alignment=TA_CENTER, spaceAfter=14),
        "section": ParagraphStyle("H", fontName="Helvetica-Bold", fontSize=11, textColor=RL_BLUE, spaceBefore=14, spaceAfter=8),
        "body":    ParagraphStyle("B", fontName="Helvetica", fontSize=10, textColor=RL_TEXT, leading=16, spaceAfter=6),
        "mono":    ParagraphStyle("M", fontName="Courier", fontSize=9, textColor=RL_MUTED, leading=14),
        "footer":  ParagraphStyle("F", fontName="Helvetica", fontSize=8, textColor=RL_MUTED, alignment=TA_CENTER),
        "verdict": ParagraphStyle("V", fontName="Helvetica-Bold", fontSize=14, textColor=rlc.white, alignment=TA_CENTER),
        "italic":  ParagraphStyle("I", fontName="Helvetica-Oblique", fontSize=10, textColor=RL_TEXT, leading=16),
    }


def _cell(text, style):
    return Paragraph(text, style)


def build_pdf(state: AuditState, output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    S = _styles()
    story = []

    metrics   = state.get("metrics", {})
    init_m    = state.get("initial_metrics", {})
    di        = float(state.get("disparate_impact_score", 0))
    init_di   = float(state.get("initial_disparate_impact", di))
    domain    = state.get("domain", "Unknown").upper()
    scols     = ", ".join(state.get("sensitive_cols", [])) or "None detected"
    narr      = state.get("explanations", {}).get("narrative", "")
    fix       = state.get("remediation_applied", "None")
    iters     = state.get("iteration_count", 0)
    shap_f    = state.get("shap_features", [])
    d_ctx     = state.get("domain_context", "")

    verdict_color = _score_color_rl(di)
    verdict_text = (
        "CRITICAL BIAS DETECTED"
        if di < 0.6 else ("WARNING — BIAS PRESENT" if di < 0.8 else "SYSTEM IS FAIR")
    )

    hdr = Table(
        [
            [Paragraph("EQUITAS AI", S["title"])],
            [Paragraph("Autonomous Bias Audit Report · Google Solution Challenge 2026", S["sub"])],
            [Paragraph(f"Domain: {domain}  ·  Sensitive Attributes: {scols}", S["mono"])],
        ],
        colWidths=[7.2 * inch],
    )
    hdr.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), RL_CARD),
        ("TOPPADDING", (0, 0), (-1, 0), 22),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 14),
        ("LEFTPADDING", (0, 0), (-1, -1), 18),
        ("RIGHTPADDING", (0, 0), (-1, -1), 18),
        ("LINEBELOW", (0, 0), (-1, 0), 2, RL_BLUE),
    ]))
    story += [hdr, Spacer(1, 0.12 * inch)]

    vrd = Table([[Paragraph(verdict_text, S["verdict"])]], colWidths=[7.2 * inch])
    vrd.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), verdict_color),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story += [vrd, Spacer(1, 0.14 * inch)]

    story.append(Paragraph("Bias Score — Before & After Remediation", S["section"]))
    gauge_buf = _buf_gauge(init_di, di)
    story.append(RLImage(gauge_buf, width=5 * inch, height=2.1 * inch))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Fairness Metrics", S["section"]))
    mh = ParagraphStyle("mh", fontName="Helvetica-Bold", fontSize=9, textColor=RL_MUTED)
    mc = ParagraphStyle("mc", fontName="Helvetica", fontSize=9, textColor=RL_TEXT)
    mm = ParagraphStyle("mm", fontName="Courier", fontSize=9, textColor=RL_MUTED)

    rows = [[_cell(h, mh) for h in ["Metric", "Before", "After", "Delta", "Status"]]]
    for k, av in metrics.items():
        if av is None:
            continue
        afv = float(av)
        bfv = float(init_m.get(k, av))
        delta = afv - bfv
        is_dir = "DIR" in k
        status = (
            "PASS" if (is_dir and afv >= 0.8) or (not is_dir and afv < 0.1)
            else "WARN" if (is_dir and afv >= 0.6) or (not is_dir and afv < 0.2)
            else "FAIL"
        )
        sc = _score_color_rl(afv) if is_dir else (RL_GREEN if afv < 0.1 else (RL_YELLOW if afv < 0.2 else RL_RED))
        rows.append([
            _cell(k, mc),
            _cell(f"{bfv:.3f}", mm),
            Paragraph(f"{afv:.3f}", ParagraphStyle("x", fontName="Courier", fontSize=9, textColor=sc)),
            Paragraph(f"{delta:+.3f}", ParagraphStyle("x", fontName="Courier", fontSize=9, textColor=RL_GREEN if delta > 0 else RL_RED)),
            Paragraph(status, ParagraphStyle("x", fontName="Helvetica-Bold", fontSize=8, textColor=sc)),
        ])

    mt = Table(rows, colWidths=[2.5 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), RL_CARD2),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [RL_CARD, RL_DARK]),
        ("LINEBELOW", (0, 0), (-1, 0), 1, RL_BLUE),
        ("GRID", (0, 0), (-1, -1), 0.3, RL_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [mt, Spacer(1, 0.12 * inch)]

    story.append(Paragraph("Metrics Comparison — Before vs After", S["section"]))
    cmp_buf = _buf_comparison(init_m, metrics)
    if cmp_buf:
        story.append(RLImage(cmp_buf, width=7 * inch, height=2.6 * inch))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("All Metrics Visualization", S["section"]))
    m_buf = _buf_metrics(metrics)
    if m_buf:
        story.append(RLImage(m_buf, width=7 * inch, height=2.8 * inch))
    story.append(Spacer(1, 0.1 * inch))

    if shap_f:
        story.append(Paragraph("Feature Importance (SHAP)", S["section"]))
        s_buf = _buf_shap(shap_f)
        if s_buf:
            story.append(RLImage(s_buf, width=6 * inch, height=2.2 * inch))
        story.append(Spacer(1, 0.1 * inch))

    if narr:
        story.append(Paragraph("AI Explainability Narrative", S["section"]))
        story.append(Paragraph(f'"{narr}"', S["italic"]))
        story.append(Spacer(1, 0.08 * inch))

    if d_ctx:
        story.append(Paragraph("Domain Legal Context", S["section"]))
        story.append(Paragraph(d_ctx, S["body"]))
        story.append(Spacer(1, 0.08 * inch))

    story.append(Paragraph("Remediation Applied", S["section"]))
    story.append(Paragraph(f"Strategy: {fix}", S["body"]))
    story.append(Paragraph(f"LangGraph iterations: {iters}", S["body"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(HRFlowable(width="100%", thickness=0.5, color=RL_BORDER))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph(
        "Generated by Equitas AI  ·  Google Solution Challenge 2026  ·  Powered by Gemini 2.0 Flash",
        S["footer"],
    ))

    doc.build(story)


async def agent_reporter(state: AuditState) -> AuditState:
    if state.get("demo_mode", False):
        await asyncio.sleep(0.4)
        return {**state, "report_path": "reports/demo_report.pdf"}

    Path("reports").mkdir(exist_ok=True)
    report_path = "reports/audit_report.pdf"
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, build_pdf, state, report_path)
    return {**state, "report_path": report_path}