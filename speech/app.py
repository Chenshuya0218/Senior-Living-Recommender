# app.py — Phase 3 final
# Streamlit app for Senior Living Recommendation
# ------------------------------------------------
# Run: streamlit run app.py
# ------------------------------------------------
import os
os.environ.setdefault("CT2_FORCE_CPU_ISA", "GENERIC")

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

# ---------- project modules ----------
# 推荐核心（你在同级目录里的 recommend_core.py）
import recommend_core as rc

# 语音转写 + 文本解析（在 ./speech/ 目录里）
try:
    from speech.transcribe import transcribe_audio
except Exception:
    transcribe_audio = None

try:
    import speech.parse_fields as pf
except Exception:
    pf = None


# -----------------------------
# Paths & data loading
# -----------------------------
HERE = Path(__file__).parent
DEFAULT_CLEAN_PATHS = [
    Path.home() / "Desktop" / "communities_clean.csv",
    HERE / "communities_clean.csv",
]
CLEAN_CSV = next((p for p in DEFAULT_CLEAN_PATHS if p.exists()), None)

st.set_page_config(page_title="Senior Living Recommender", layout="wide")
st.title("🏡 Senior Living Recommender")

if CLEAN_CSV is None:
    st.error(
        "Could not find **communities_clean.csv**.\n\n"
        "Please run Phase-1 to produce it and place the file on Desktop or next to app.py."
    )
    st.stop()

# ✅ 统一：使用 recommend_core.load_clean() 加载并清洗
try:
    df_reco = rc.load_clean(str(CLEAN_CSV))
except Exception as e:
    st.error(f"Failed to load clean CSV via rc.load_clean(): {e}")
    st.stop()

# 一点数据可用性提示
usable = pd.to_numeric(df_reco.get("price", np.nan), errors="coerce").notna().sum()
st.caption(f"Records used for recommendation: **{usable}** (file: {CLEAN_CSV.name})")


# -----------------------------
# Sidebar: manual profile
# -----------------------------
st.sidebar.header("Client Profile (manual)")

budget = st.sidebar.number_input("Monthly Budget ($)", min_value=0, value=4000, step=50)

service_options = [
    "(any)",
    "assisted living",
    "independent living",
    "memory care",
    "enhanced assisted living",
    "nursing care",
]
preferred_service = st.sidebar.selectbox("Preferred Service", service_options, index=1)

needs_pick = st.sidebar.multiselect(
    "Select Needs",
    ["wheelchair accessible", "medication support", "mobility support", "diaper support"],
    default=[],
)
preferred_city = st.sidebar.text_input("Preferred City (optional)")
preferred_state = st.sidebar.text_input("Preferred State (optional)").upper()

# 权重（可选）
with st.sidebar.expander("Advanced Weights (optional)", expanded=False):
    W_BUDGET_IN   = st.slider("Weight: within budget", 0.0, 5.0, 2.0, 0.1)
    W_BUDGET_FUZZ = st.slider("Weight: slightly above budget (≤10%)", 0.0, 5.0, 1.0, 0.1)
    W_SERVICE     = st.slider("Weight: service match", 0.0, 5.0, 3.0, 0.1)
    W_CITY        = st.slider("Weight: city match", 0.0, 5.0, 1.0, 0.1)
    W_STATE       = st.slider("Weight: state match", 0.0, 5.0, 1.0, 0.1)
    W_NEED        = st.slider("Weight: per-need match", 0.0, 5.0, 1.0, 0.1)
    W_PARTNER     = st.slider("Weight: partner priority", 0.0, 5.0, 0.5, 0.1)
    W_GEO         = st.slider("Weight: geo points", 0.0, 5.0, 0.3, 0.1)

weights: Dict[str, float] = dict(
    W_BUDGET_IN=W_BUDGET_IN,
    W_BUDGET_FUZZ=W_BUDGET_FUZZ,
    W_SERVICE=W_SERVICE,
    W_CITY=W_CITY,
    W_STATE=W_STATE,
    W_NEED=W_NEED,
    W_PARTNER=W_PARTNER,
    W_GEO=W_GEO,
)


# -----------------------------
# Sidebar: voice/text intake
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Voice/Text Intake")

audio_file = st.sidebar.file_uploader(
    "Upload call recording (.m4a/.mp3/.wav)", type=["m4a", "mp3", "wav"]
)
raw_text = st.sidebar.text_area("Or paste intake text here")

auto_btn = st.sidebar.button("Transcribe & Parse")

# 用于保存最近一次解析的用户画像，便于用户对照/修正
if "parsed_profile" not in st.session_state:
    st.session_state.parsed_profile = None

if auto_btn:
    parsed = None
    try:
        text = None
        if audio_file is not None:
            if transcribe_audio is None:
                st.error("Transcription is unavailable (module not imported).")
            else:
                import tempfile

                suffix = f".{audio_file.name.split('.')[-1]}"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name

                text, _ = transcribe_audio(tmp_path, model_size="base")
                os.unlink(tmp_path)  # 删除临时文件
        elif raw_text and raw_text.strip():
            text = raw_text.strip()
        else:
            st.warning("Please upload an audio file or paste text.")

        if text:
            if pf is None:
                st.error("Parser is unavailable (module not imported).")
            else:
                parsed = pf.extract_fields(text)

                # 安全兜底
                parsed.setdefault("preferred_city", "")
                parsed.setdefault("preferred_state", "")
                parsed.setdefault("budget", None)
                parsed.setdefault("needs", {})
                for k in ["wheelchair_accessible", "medication_support", "diaper_support", "mobility_support"]:
                    parsed["needs"].setdefault(k, False)

                st.session_state.parsed_profile = parsed

                st.success("Parsed from voice/text:")
                st.json(parsed)

                # 直接预览推荐（不覆盖手动输入控件）
                preview = rc.recommend(df_reco, parsed, top_k=10, weights=weights)
                st.caption("Preview recommendations based on parsed profile:")
                st.dataframe(rc.pretty_results(preview), use_container_width=True)
    except Exception as e:
        st.error(f"Transcribe/parse failed: {e}")


# -----------------------------
# Build client dict (from manual controls)
# -----------------------------
client = {
    "budget": float(budget) if budget else None,
    "preferred_service": "" if preferred_service == "(any)" else preferred_service,
    "preferred_city": preferred_city,
    "preferred_state": preferred_state,
    "needs": {
        "wheelchair_accessible": "wheelchair accessible" in needs_pick,
        "medication_support": "medication support" in needs_pick,
        "mobility_support": "mobility support" in needs_pick,
        "diaper_support": "diaper support" in needs_pick,
    },
}

# 让用户选择是否直接使用“解析得到的画像”
use_parsed = False
if st.session_state.parsed_profile:
    use_parsed = st.toggle("Use last parsed profile (override manual inputs)", value=False)
if use_parsed:
    client = st.session_state.parsed_profile


# -----------------------------
# Main: filter (optional) + recommend
# -----------------------------
st.header("Top Recommendations")

go = st.sidebar.button("Generate Recommendations", type="primary")

if go:
    # 轻度前置过滤（非硬过滤，便于聚焦）
    work = df_reco.copy()
    if client.get("preferred_service"):
        work = work[work["services_offered"].fillna("").str.contains(client["preferred_service"], na=False)]
    if client.get("preferred_city"):
        work = work[work["city"].str.lower().eq(client["preferred_city"].lower())]
    if client.get("preferred_state"):
        work = work[work["state"].str.upper().eq(client["preferred_state"].upper())]

    results = rc.recommend(work, client, top_k=10, weights=weights)
    pretty = rc.pretty_results(results)
    st.dataframe(pretty, use_container_width=True)

    # 下载
    st.download_button(
        "Download CSV",
        data=pretty.to_csv(index=False).encode("utf-8"),
        file_name="recommendations.csv",
        mime="text/csv",
    )
else:
    st.info("Set profile on the left, then click **Generate Recommendations**.")


# -----------------------------
# Debug / Peek
# -----------------------------
with st.sidebar.expander("Debug / Data peek", expanded=False):
    st.write("Clean file:", str(CLEAN_CSV))
    st.write("Data shape:", df_reco.shape)
    st.dataframe(df_reco.head(8))
