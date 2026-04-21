# Equitas AI — Autonomous Bias Detection & Remediation Platform

> **Google Solution Challenge 2026 · Unbiased AI Decision Track**  
> Built by a solo developer using Gemini AI, LangGraph, and FastAPI.

---

## What is Equitas AI?

Equitas AI is an autonomous fairness auditing platform that detects, explains, and fixes hidden bias in AI datasets and models — before they harm real people.

Computer programs now make life-changing decisions about who gets a job, a bank loan, or medical care. If these programs learn from flawed historical data, they repeat and amplify those exact same discriminatory mistakes.

**Equitas AI solves this** with a 5-agent Gemini-powered pipeline that autonomously audits a dataset, explains the bias in plain English, applies a fix, then re-audits to verify — all in one loop.

---

## Live Demo

```
Frontend : open index.html in browser
Backend  : uvicorn main:app --reload --port 8000
```

---

## Key Features

| Feature | Description |
|---|---|
| **5-Agent LangGraph Pipeline** | Profiler → Detector → Remediator → Explainer → Reporter running autonomously |
| **5 Fairness Metrics** | Disparate Impact, Demographic Parity, Equal Opportunity, Equalized Odds, Predictive Parity |
| **SHAP Explainability** | Feature importance charts + Gemini plain-English narration |
| **Intersectional Heatmap** | Race × Gender approval rate matrix |
| **Regulatory Compliance Badges** | Auto-detects EEOC 80% Rule, ECOA, Civil Rights Act violations |
| **Remediation Agent** | Reweighing / resampling strategies with before/after comparison |
| **What-If Simulator** | Drag a slider to see how changing group ratios affects DIR score |
| **FairBot AI Chat** | Ask anything about your audit — answered by Gemini with full context |
| **Model Card Generator** | Google-standard Model Card auto-generated from audit results |
| **Styled PDF Report** | Charts, metrics table, gauge, narrative — downloadable |
| **Bias Trend History** | Line chart of DIR scores across all past audits |
| **Demo Mode** | Instant cached results for presentations — no wait time |
| **Sample Datasets** | COMPAS, Adult Income, German Credit — one-click load |
| **Welcome Modal + Tooltips** | Onboarding overlay + hover tooltips on every nav item |

---

## System Architecture

```
User (Browser)
    │
    ▼
index.html  ──────────────────────────────────────────────────────
  9 Views: Upload · Audit · Dashboard · Explainability ·
           Remediation · Report · FairBot · Model Card · What-If
    │
    ▼ HTTP / SSE
FastAPI (main.py)
  ├── POST /upload          → Save CSV, return preview
  ├── GET  /audit/stream    → SSE: stream agent events live
  ├── GET  /history         → Past audit JSON records
  ├── GET  /sample/{name}   → Load built-in sample dataset
  ├── POST /ask             → FairBot: Gemini answers questions
  ├── POST /model-card      → Generate Google Model Card
  ├── POST /what-if         → Simulate group ratio changes
  └── GET  /download-report → Serve styled PDF
    │
    ▼ LangGraph StateGraph
Agent Pipeline
  1. Profiler    → Detect sensitive columns, domain, legal context
  2. Detector    → Compute 5 fairness metrics + intersectional matrix
  3. Remediator  → JSON RAG lookup + Gemini fix recommendation
  4. Explainer   → SHAP values + Gemini counterfactual narrative
  5. Reporter    → ReportLab PDF with matplotlib charts
    │
    ▼ Decision Gate
  DIR < 0.8 AND iterations < 2 → loop back to Detector
  DIR ≥ 0.8 OR max iterations  → proceed to Explainer
```

---

## Agentic Loop

```
Upload Dataset
      ↓
  Agent 1: Profiler
  - Gemini detects sensitive cols (race, gender, age...)
  - Tags domain: hiring / lending / healthcare / criminal justice
      ↓
  Agent 2: Bias Detector
  - Disparate Impact Ratio
  - Demographic Parity Difference
  - Equal Opportunity Difference
  - Equalized Odds Difference
  - Predictive Parity Difference
      ↓
  ┌── DIR < 0.8? ──┐
  │                │ YES (bias found)
  │         Agent 3: Remediator
  │         - Query past audit history (JSON RAG)
  │         - Gemini recommends fix strategy
  │         - Apply reweighing / resampling
  │                │
  └────────────────┘ (re-audit, max 2 loops)
      ↓ DIR ≥ 0.8 or max iterations
  Agent 4: Explainer
  - SHAP feature importance (500-row sample)
  - Gemini generates plain-English counterfactual
      ↓
  Agent 5: Reporter
  - ReportLab PDF: gauge + bar charts + metrics table
  - Audit saved to history (JSON RAG memory)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI Agents | Google Gemini 2.5 Flash Lite |
| Agentic Framework | LangGraph (StateGraph) |
| Fairness Metrics | AIF360, Fairlearn, SHAP |
| Backend | FastAPI, Python 3.11, asyncio |
| Frontend | Vanilla HTML/CSS/JS, Chart.js, GSAP |
| PDF Generation | ReportLab + Matplotlib |
| Deployment | Google Cloud Run ready |

---

## Project Structure

```
equitas_ai/
├── main.py                  # FastAPI app, LangGraph wiring, all endpoints
├── index.html               # Single-file frontend — 9 views, JS routing
├── requirements.txt
├── .env                     # GEMINI_API_KEY
├── agents/
│   ├── state.py             # LangGraph TypedDict state
│   ├── profiler.py          # Agent 1 — domain + sensitive col detection
│   ├── detector.py          # Agent 2 — 5 fairness metrics
│   ├── explainer.py         # Agent 3 — SHAP + Gemini narration
│   ├── remediator.py        # Agent 4 — RAG + fix strategy
│   └── reporter.py          # Agent 5 — styled PDF generation
├── sample_data/
│   ├── compas.csv           # Criminal justice dataset
│   ├── adult_income.csv     # Hiring / income dataset
│   └── german_credit.csv    # Lending dataset
├── audit_history/           # JSON RAG memory (auto-created)
├── uploads/                 # User uploaded datasets (auto-created)
└── reports/                 # Generated PDFs (auto-created)
```

---

# 🚀 Setup & Run

### A. Local Setup (AI Studio)
```bash
# 1. Clone & Enter
git clone https://github.com/vansh7266/google-solution-challenge-2026
cd google-solution-challenge-2026/equitas_ai

# 2. Environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. Launch
uvicorn main:app --reload --port 8000
open index.html
```

### B. Cloud Deployment (Vertex AI)
Equitas AI is enterprise-ready and optimized for **Google Cloud Run**. It utilizes **Vertex AI** to leverage high-throughput scaling and GCP credits.

```bash
# 1. Configure GCP
gcloud auth login
gcloud config set project poetic-standard-494013-j6

# 2. Enable Vertex & Cloud Run APIs
gcloud services enable run.googleapis.com aiplatform.googleapis.com

# 3. Deploy (Keyless IAM)
gcloud run deploy equitas-ai --source . --region us-central1 --allow-unauthenticated
```
*Note: The system automatically detects the Cloud environment and switches to Vertex AI (IAM-based) from AI Studio (Key-based) using the built-in `ai_config.py` utility.*

---

## Fairness Metrics Explained

| Metric | Threshold | Meaning |
|---|---|---|
| **Disparate Impact Ratio** | ≥ 0.80 | EEOC 80% rule — ratio of approval rates between groups |
| **Demographic Parity Diff** | < 0.10 | Difference in positive prediction rates |
| **Equal Opportunity Diff** | < 0.10 | Difference in true positive rates |
| **Equalized Odds Diff** | < 0.10 | Combined TPR + FPR difference |
| **Predictive Parity Diff** | < 0.10 | Difference in precision across groups |

---

## Sample Datasets

| Dataset | Domain | Bias Type |
|---|---|---|
| **COMPAS** | Criminal Justice | Racial bias in recidivism prediction |
| **Adult Income (UCI)** | Hiring | Gender bias in income prediction |
| **German Credit** | Lending | Age/gender bias in credit decisions |

---

## Regulatory Compliance Detection

Equitas AI automatically maps detected bias to violated regulations:

- **Criminal Justice** → Equal Protection Clause, Civil Rights Act Title VI, EEOC 80% Rule
- **Hiring** → EEOC Uniform Guidelines, Civil Rights Act Title VII, ADEA
- **Lending** → Equal Credit Opportunity Act, Fair Housing Act, CFPB Rules
- **Healthcare** → ACA Section 1557, Americans with Disabilities Act, HIPAA
- **Education** → Title IX, Title VI, IDEA

---

## Problem Statement

**[Unbiased AI Decision] Ensuring Fairness and Detecting Bias in Automated Decisions**

> Build a clear, accessible solution to thoroughly inspect datasets and software models for hidden unfairness or discrimination. Provide organizations with an easy way to measure, flag, and fix harmful bias before their systems impact real people.

**Google Solution Challenge 2026 · India · Build with AI**

---

## Demo Datasets

For judges and evaluators — use Demo Mode (top-right button) for instant results without API calls, or load sample datasets with one click on the Upload page.

---

*Built with Gemini AI · Google Solution Challenge 2026*