# recommend_core.py
# ------------------------------------------------------------
# Core data prep + ranking logic used by Phase-2 / main pipeline
# ------------------------------------------------------------

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# -----------------------------
# Public constants
# -----------------------------
NEED_COLUMNS: List[str] = [
    "wheelchair_accessible",
    "medication_support",
    "diaper_support",
    "mobility_support",
]

# 可选：服务类型归一化（和 app.py/Phase-2 保持一致）
SERVICE_CANON: Dict[str, str] = {
    r"(?:assisted\s*living|\bassisted\b)": "assisted living",
    r"(?:independent|indep|active\s*(?:adult|senior)|senior\s*apart)": "independent living",
    r"(?:memory|alz|dement)": "memory care",
    r"(?:enhanced|enriched)": "enhanced assisted living",
    r"(?:skilled\s*nursing|nursing\s*home|\bsnf\b|rehab)": "nursing care",
}

# 默认权重（可被 recommend(..., weights=...) 覆盖）
DEFAULT_WEIGHTS = dict(
    W_BUDGET_IN=2.0,      # 价格 <= 预算
    W_BUDGET_FUZZ=1.0,    # 价格略高（<=10%）软加分
    W_SERVICE=3.0,        # 服务类型匹配
    W_CITY=1.0,           # 城市匹配
    W_STATE=1.0,          # 州匹配
    W_NEED=1.0,           # 每个需求命中
    W_PARTNER=0.5,        # 合作优先（若有 partner_priority 列）
    W_GEO=0.3,            # 地缘点（若有 geo_points 列）
)

# -----------------------------
# Internal helpers
# -----------------------------
def _to_bool_series(x: pd.Series) -> pd.Series:
    """把看起来像真假值的文本转成 True/False/NA（pandas nullable bool）"""
    s = x.astype(str).str.strip().str.lower()
    out = s.map({
        "true": True, "t": True, "1": True, "yes": True, "y": True,
        "false": False, "f": False, "0": False, "no": False, "n": False,
    })
    return out.astype("boolean")

def _canonical_service(s: pd.Series) -> pd.Series:
    """服务类型归一化：正则匹配到统一标签；其余保持原值的小写"""
    base = s.astype(str).str.lower()
    out = base.copy()
    for pat, lab in SERVICE_CANON.items():
        mask = base.str.contains(pat, na=False)
        out = out.mask(mask, lab)
    return out

# -----------------------------
# public: data prep
# -----------------------------
def prep_for_reco(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Make the community dataframe safe for scoring.
    - 去重重复列名
    - price -> numeric
    - 四个 need 列转为 True/False/<NA>
    - city/state 兜底为 ""，并做大小写规范
    - services_offered 归一化
    - partner_priority / geo_points 若存在，转为 int
    """
    df = df_in.copy()

    # 去重重复列名（保留第一个）
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()

    # price -> numeric
    if "price" not in df.columns:
        df["price"] = np.nan
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # need 列 -> boolean
    for c in NEED_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = _to_bool_series(df[c])

    # city/state
    for c in ["city", "state"]:
        if c not in df.columns:
            df[c] = ""
    df["city"] = (
        df["city"].astype(str)
        .replace(["nan", "None"], "")
        .fillna("")
        .str.title()
    )
    df["state"] = (
        df["state"].astype(str)
        .replace(["nan", "None"], "")
        .fillna("")
        .str.upper()
    )

    # community_name 友好显示
    if "community_name" not in df.columns:
        df["community_name"] = ""
    df["community_name"] = (
        df["community_name"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .fillna("")
    )

    # 服务类型统一小写+归一化
    if "services_offered" not in df.columns:
        df["services_offered"] = ""
    df["services_offered"] = _canonical_service(df["services_offered"])

    # 可选辅助列
    if "_partner_pri_" in df.columns and "partner_priority" not in df.columns:
        df = df.rename(columns={"_partner_pri_": "partner_priority"})
    if "partner_priority" in df.columns:
        df["partner_priority"] = pd.to_numeric(df["partner_priority"], errors="coerce").fillna(0).astype(int)

    if "geo_points" in df.columns:
        df["geo_points"] = pd.to_numeric(df["geo_points"], errors="coerce").fillna(0).astype(int)

    return df

def load_clean(clean_csv: str) -> pd.DataFrame:
    """
    读取 Phase-1 生成的 communities_clean.csv 并做二次兜底清洗。
    这是 main_pipeline / App 里推荐使用的入口。
    """
    df_raw = pd.read_csv(clean_csv, dtype=str)
    return prep_for_reco(df_raw)

# -----------------------------
# public: ranking
# -----------------------------
def recommend(
    df_reco: pd.DataFrame,
    client: Dict,
    top_k: int = 10,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    给定准备好的 df_reco 和 client already-parsed dict:
      client = {
        "budget": 4000,
        "preferred_service": "assisted living",
        "preferred_city": "Rochester",
        "preferred_state": "NY",
        "needs": {
            "wheelchair_accessible": True/False,
            "medication_support": True/False,
            "mobility_support": True/False,
            "diaper_support": True/False
        }
      }
    返回按分数排序的 TopK。
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    # 取权重（避免缺键）
    W_BUDGET_IN   = float(weights.get("W_BUDGET_IN",   DEFAULT_WEIGHTS["W_BUDGET_IN"]))
    W_BUDGET_FUZZ = float(weights.get("W_BUDGET_FUZZ", DEFAULT_WEIGHTS["W_BUDGET_FUZZ"]))
    W_SERVICE     = float(weights.get("W_SERVICE",     DEFAULT_WEIGHTS["W_SERVICE"]))
    W_CITY        = float(weights.get("W_CITY",        DEFAULT_WEIGHTS["W_CITY"]))
    W_STATE       = float(weights.get("W_STATE",       DEFAULT_WEIGHTS["W_STATE"]))
    W_NEED        = float(weights.get("W_NEED",        DEFAULT_WEIGHTS["W_NEED"]))
    W_PARTNER     = float(weights.get("W_PARTNER",     DEFAULT_WEIGHTS["W_PARTNER"]))
    W_GEO         = float(weights.get("W_GEO",         DEFAULT_WEIGHTS["W_GEO"]))

    df = df_reco.copy()

    budget  = client.get("budget")
    svc     = (client.get("preferred_service") or "").strip().lower()
    city    = (client.get("preferred_city") or "").strip().lower()
    state   = (client.get("preferred_state") or "").strip().upper()
    needs   = client.get("needs", {}) or {}

    scores: List[float] = []
    whys: List[str] = []

    for _, r in df.iterrows():
        score = 0.0
        why_parts: List[str] = []

        # price
        price = r.get("price", np.nan)
        if pd.notna(price) and pd.notna(budget):
            try:
                price_f = float(price)
                if price_f <= float(budget):
                    score += W_BUDGET_IN
                    why_parts.append("within budget")
                elif price_f <= float(budget) * 1.10:
                    score += W_BUDGET_FUZZ
                    why_parts.append("slightly above budget (≤10%)")
                else:
                    why_parts.append("over budget")
            except Exception:
                pass

        # service
        ro_svc = str(r.get("services_offered", "")).strip().lower()
        if svc and ro_svc:
            if svc in ro_svc:
                score += W_SERVICE
                why_parts.append("service matches")

        # city/state
        ro_city  = str(r.get("city", "")).strip().lower()
        ro_state = str(r.get("state", "")).strip().upper()
        if city and ro_city and (ro_city == city):
            score += W_CITY
            why_parts.append("city matches")
        if state and ro_state and (ro_state == state):
            score += W_STATE
            why_parts.append("state matches")

        # needs
        need_hits, need_unknown = [], []
        for need_col, want in (needs or {}).items():
            if not want:
                continue
            val = r.get(need_col, pd.NA)
            if pd.isna(val):
                need_unknown.append(need_col)
            elif bool(val) is True:
                score += W_NEED
                need_hits.append(need_col)
        if need_hits:
            why_parts.append("needs matched: " + ", ".join(need_hits))
        if need_unknown:
            why_parts.append("needs unknown: " + ", ".join(need_unknown))

        # partner / geo（如果列存在）
        if "partner_priority" in df.columns:
            pri = r.get("partner_priority", 0)
            if pd.notna(pri) and pri:
                score += W_PARTNER * float(pri)
        if "geo_points" in df.columns:
            gp = r.get("geo_points", 0)
            if pd.notna(gp) and gp:
                score += W_GEO * float(gp)

        scores.append(score)
        whys.append("; ".join(why_parts) if why_parts else "")

    df["score"] = scores
    df["why"] = whys

    # 排序：合作优先 -> 分数高 -> 价格低
    sort_cols: List[str] = []
    asc: List[bool] = []
    if "partner_priority" in df.columns:
        sort_cols.append("partner_priority"); asc.append(False)
    sort_cols += ["score", "price"]; asc += [False, True]

    df0 = df.sort_values(by=sort_cols, ascending=asc)

    out_cols = [
        "community_name","services_offered","price","city","state",
        "wheelchair_accessible","medication_support","diaper_support","mobility_support",
        "partner_priority","geo_points","score","why"
    ]
    out_cols = [c for c in out_cols if c in df0.columns]
    return df0[out_cols].head(int(top_k)).reset_index(drop=True)

# -----------------------------
# public: pretty output
# -----------------------------
def pretty_results(df: pd.DataFrame) -> pd.DataFrame:
    """把 NA 显示为 <NA>，便于导出/展示"""
    out = df.copy()
    for c in NEED_COLUMNS:
        if c in out.columns:
            out[c] = out[c].astype(object).where(~out[c].isna(), "<NA>")
    return out

__all__ = [
    "NEED_COLUMNS",
    "SERVICE_CANON",
    "DEFAULT_WEIGHTS",
    "prep_for_reco",
    "load_clean",
    "recommend",
    "pretty_results",
]
