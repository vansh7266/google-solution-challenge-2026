# --- SECTION: Imports ---
from typing import TypedDict, List, Dict, Any

# --- SECTION: State Definition ---
class AuditState(TypedDict):
    dataset_path: str
    domain: str
    sensitive_cols: List[str]
    metrics: Dict[str, Any]
    disparate_impact_score: float
    explanations: Dict[str, Any]
    remediation_applied: str
    iteration_count: int
    demo_mode: bool