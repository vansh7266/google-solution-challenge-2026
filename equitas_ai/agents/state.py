# --- SECTION: Imports ---
from typing import TypedDict, List, Dict, Any, Optional

# --- SECTION: State Definition ---
class AuditState(TypedDict):
    # Input
    dataset_path:             str
    demo_mode:                bool

    # Profiler output
    domain:                   str
    domain_context:           str
    sensitive_cols:           List[str]

    # Detector output — stores BOTH initial and current metrics
    metrics:                  Dict[str, Any]
    initial_metrics:          Dict[str, Any]
    disparate_impact_score:   float
    initial_disparate_impact: float
    intersectional_matrix:    Dict[str, Any]

    # Explainer output
    shap_features:            List[Dict[str, Any]]
    explanations:             Dict[str, Any]

    # Remediator output
    remediation_applied:      str
    iteration_count:          int

    # Reporter output
    report_path:              str