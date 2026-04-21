import pandas as pd
import numpy as np
import asyncio
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from .state import AuditState


def _binarize(s: pd.Series, domain: str = "general") -> np.ndarray:
    if s.dtype == object or str(s.dtype) == "category":
        enc = LabelEncoder().fit_transform(s.astype(str))
        target_val = enc.min() if domain == "criminal_justice" else enc.max()
        return (enc == target_val).astype(int)
    
    # Numerical target handling
    if domain == "criminal_justice":
        # In COMPAS, 0 is 'no recidivism', which is the good outcome.
        return (s == 0).astype(int)
    return (s > s.median()).astype(int)


def _dir_score(df: pd.DataFrame, sensitive_col: str, target_col: str) -> float:
    y = _binarize(df[target_col])
    tmp = df.copy()
    tmp["_y"] = y
    rates = tmp.groupby(sensitive_col)["_y"].mean()
    if len(rates) < 2 or rates.max() == 0:
        return 1.0
    return round(float(rates.min() / rates.max()), 3)


def _model_metrics(df: pd.DataFrame, sensitive_col: str, target_col: str) -> dict:
    sample = df.sample(n=min(2000, len(df)), random_state=42).dropna()
    X = sample.drop(columns=[target_col])
    y = _binarize(sample[target_col])
    X_enc = X.apply(lambda c: LabelEncoder().fit_transform(c.astype(str)))
    clf = LogisticRegression(max_iter=300, C=0.1, random_state=42)
    clf.fit(X_enc, y)
    y_pred = clf.predict(X_enc)
    sf = sample[sensitive_col].astype(str)

    from fairlearn.metrics import (
        demographic_parity_difference,
        equalized_odds_difference,
        equal_opportunity_difference,
    )

    dp = round(float(abs(demographic_parity_difference(y, y_pred, sensitive_features=sf))), 3)
    eo = round(float(abs(equal_opportunity_difference(y, y_pred, sensitive_features=sf))), 3)
    eod = round(float(abs(equalized_odds_difference(y, y_pred, sensitive_features=sf))), 3)

    groups = sf.unique()
    pps = {
        g: precision_score(y[sf == g], y_pred[sf == g], zero_division=0)
        for g in groups
        if (sf == g).sum() > 5
    }
    pp = round(max(pps.values()) - min(pps.values()), 3) if len(pps) >= 2 else 0.0

    return {"dp": dp, "eo": eo, "eod": eod, "pp": pp}


async def agent_bias_detector(state: AuditState) -> AuditState:
    is_first = state.get("iteration_count", 0) == 0

    if state.get("demo_mode", False):
        await asyncio.sleep(0.5)
        demo = {
            "Disparate Impact (DIR)": 0.64,
            "Demographic Parity Diff": 0.21,
            "Equal Opportunity Diff": 0.18,
            "Equalized Odds Diff": 0.23,
            "Predictive Parity Diff": 0.15,
        }
        inter = {
            "White+Male": 0.89,
            "White+Female": 0.76,
            "Black+Male": 0.58,
            "Black+Female": 0.41,
        }
        return {
            **state,
            "disparate_impact_score": 0.64,
            "initial_disparate_impact": 0.64 if is_first else state.get("initial_disparate_impact", 0.64),
            "metrics": demo,
            "initial_metrics": demo if is_first else state.get("initial_metrics", demo),
            "intersectional_matrix": inter,
        }

    df = pd.read_csv(state["dataset_path"])
    sensitive_cols = state.get("sensitive_cols", [])
    target_col = df.columns[-1]

    if not sensitive_cols:
        empty = {
            "Disparate Impact (DIR)": 1.0,
            "Demographic Parity Diff": 0.0,
            "Equal Opportunity Diff": 0.0,
            "Equalized Odds Diff": 0.0,
            "Predictive Parity Diff": 0.0,
        }
        return {
            **state,
            "disparate_impact_score": 1.0,
            "initial_disparate_impact": 1.0,
            "metrics": empty,
            "initial_metrics": empty,
            "intersectional_matrix": {},
        }

    sensitive_col = sensitive_cols[0]
    domain = state.get("domain", "general")
    di = _dir_score(df, sensitive_col, target_col) # Note: _dir_score needs domain too, updating it
    y_bin = _binarize(df[target_col], domain)
    
    # Internal function update to avoid signature change drama
    def _get_dir(sub_df, s_col, t_col):
        rates = sub_df.groupby(s_col)["_y"].mean()
        if len(rates) < 2 or rates.max() == 0: return 1.0
        return round(float(rates.min() / rates.max()), 3)

    tmp_df = df.copy()
    tmp_df["_y"] = y_bin
    di = _get_dir(tmp_df, sensitive_col, target_col)

    try:
        m = _model_metrics(df, sensitive_col, target_col)
    except Exception:
        m = {"dp": 0.0, "eo": 0.0, "eod": 0.0, "pp": 0.0}

    metrics = {
        "Disparate Impact (DIR)": di,
        "Demographic Parity Diff": m["dp"],
        "Equal Opportunity Diff": m["eo"],
        "Equalized Odds Diff": m["eod"],
        "Predictive Parity Diff": m["pp"],
    }

    inter = {}
    if len(sensitive_cols) >= 2:
        try:
            col2 = sensitive_cols[1]
            tmp = df.copy()
            tmp["_y"] = _binarize(df[target_col], domain)
            
            # Smart Binning for numerical columns (like age)
            if pd.api.types.is_numeric_dtype(tmp[col2]) and tmp[col2].nunique() > 10:
                tmp[col2] = pd.cut(tmp[col2], bins=[0, 25, 45, 100], labels=["<25", "25-45", "45+"])
            
            mat = tmp.groupby([sensitive_col, col2])["_y"].mean()
            inter = {f"{k[0]}+{k[1]}": round(float(v), 3) for k, v in mat.items()}
        except Exception:
            pass

    result = {
        **state,
        "disparate_impact_score": di,
        "metrics": metrics,
        "intersectional_matrix": inter,
    }

    if is_first:
        result["initial_disparate_impact"] = di
        result["initial_metrics"] = metrics

    return result