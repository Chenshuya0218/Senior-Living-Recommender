import re

# ========== 服务类型（用于识别 assisted / independent / nursing 等） ==========
SERVICE_CANON = {
    r"(?:(assisted)\s+living|bassist?ed\b)": "assisted living",
    r"(?:independent|active\s*(?:adult|senior)|senior\s*apart)": "independent living",
    r"(?:memory|alz|dement)": "memory care",
    r"(?:enhanced|enriched)": "enhanced assisted living",
    r"(?:skilled\s*nursing|nursing\s*home|rehab|bsnf\b)": "nursing care",
}

# ========== 核心函数：提取客户画像信息 ==========
def extract_fields(text: str) -> dict:
    """
    从转写文本中提取关键信息：预算、服务类型、城市州、需求标签
    返回一个结构化字典
    """
    if not text:
        return {}

    t = text.lower()

    # ========== 1️⃣ 预算 budget ==========
    budget = _parse_budget(t)

    # ========== 2️⃣ 服务类型 service ==========
    service = _parse_service(t)

    # ========== 3️⃣ 城市与州 ==========
    city, state = _parse_city_state(text)

    # ========== 4️⃣ 客户需求 ==========
    needs = _parse_needs(t)

    return {
        "budget": budget,
        "preferred_service": service,
        "preferred_city": city,
        "preferred_state": state,
        "needs": needs,
    }

# ---------- 解析预算 ----------
def _parse_budget(t: str):
    m = re.search(r"\$?\s*([0-9][\d,\.]*)\s*(?:/|per)?\s*(?:month|mo|mth)?", t)
    if not m:
        return None
    try:
        return float(re.sub(r"[^\d\.]", "", m.group(1)))
    except Exception:
        return None

# ---------- 解析服务类型 ----------
def _parse_service(t: str):
    for pat, label in SERVICE_CANON.items():
        if re.search(pat, t, flags=re.I):
            return label
    return None

# ---------- 改进版：解析城市 + 州 ----------
def _parse_city_state(text: str):
    """
    尝试从文本中提取城市(city)和州(state)信息。
    更智能地过滤掉“in St. Anne's”、“in rehab”、“in hospital”等非地名。
    """
    t = text.lower()

    # 匹配常见格式: in Penfield, NY / in Rochester NY / in Penfield New York
    city_state_pattern = re.search(
        r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)[, ]+\s*([A-Z]{2}|new\s+york|california|florida|texas|indiana|ohio|illinois|michigan|pennsylvania)\b",
        text,
        flags=re.I
    )

    # 提取城市和州
    if city_state_pattern:
        city = city_state_pattern.group(1).strip().title()
        state = city_state_pattern.group(2).strip().upper()[:2]
    else:
        city, state = None, None

    # 排除“St. Anne’s”、“rehab”、“hospital”等误识别为地名的情况
    if city and any(x in city.lower() for x in ["rehab", "hospital", "home", "center", "st. anne", "facility", "community"]):
        city = None

    # 排除单独“in”被误判为“IN”州缩写的情况
    if state == "IN" and not city:
        state = None

    return city, state

# ---------- 解析需求 ----------
def _parse_needs(t: str) -> dict:
    need_tags = {
        "wheelchair_accessible": ["wheelchair", "walker", "accessible", "mobility"],
        "medication_support": ["medication", "pharmacy", "nurse help", "medical"],
        "diaper_support": ["continence", "incontinent", "diaper"],
        "mobility_support": ["mobility", "transfer", "walker", "cane"],
    }

    needs = {}
    for key, patterns in need_tags.items():
        needs[key] = any(re.search(p, t) for p in patterns)

    return needs


# ---------- 测试函数（可选） ----------
if __name__ == "__main__":
    sample_text = """
    Hi, this is a referral from St. Anne's rehab.
    Margaret Thompson lives in Penfield, NY.
    She's looking for enhanced assisted living, budget around $5500 per month.
    She uses a walker but doesn't need medication help.
    """

    parsed = extract_fields(sample_text)
    print(parsed)
