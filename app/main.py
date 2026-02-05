# =========================
# Path setup (MUST be first)
# =========================
import sys
import os
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from services.user_insights import (
    get_user_or_case_insights,
    get_pending_cases,
    get_overdue_cases,
    get_critical_cases
)

# =========================
# Imports
# =========================
import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from services.retriever import find_similar_cases

# =========================
# Streamlit UI Config
# =========================
st.set_page_config(
    page_title="Auto MPR Recommendation",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Load Custom CSS
# =========================
def load_css():
    css_path = Path(__file__).parent.parent / "styles" / "ui.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =========================
# Helper: Build Recommendation (PDF + CSV SAFE)
# =========================
def build_recommendation(results, top_n=3):
    recommendations = []

    POSSIBLE_TEXT_FIELDS = [
        "Resolution",
        "resolution",
        "solution",
        "answer",
        "details",
        "text",
        "content",
        "page_content",
        "chunk"
    ]

    for r in results[:top_n]:
        for field in POSSIBLE_TEXT_FIELDS:
            if r.get(field):
                recommendations.append(f"• {r[field]}")
                break

    if not recommendations:
        return "No clear resolution found in historical cases."

    return "\n".join(recommendations)

# =========================
# Header (Logo + Mode)
# =========================
with st.container():
    _, logo_col, _ = st.columns([1, 1, 1])
    with logo_col:
        st.image("assets/logo.png", width=280)

    _, radio_col, _ = st.columns([1, 1, 1])
    with radio_col:
        query_mode = st.radio(
            "",
            ["General MPR Issue", "User-Specific View"],
            horizontal=True,
            label_visibility="collapsed"
        )

# =========================
# Reset state on mode change
# =========================
if "last_query_mode" not in st.session_state:
    st.session_state.last_query_mode = query_mode

if st.session_state.last_query_mode != query_mode:
    st.session_state.user_summary = None
    st.session_state.active_owner = None
    st.session_state.last_query_mode = query_mode

# =========================
# Load FAISS + Model
# =========================
@st.cache_resource
def load_resources():
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = BASE_DIR / "data"

    index_path = DATA_DIR / "pdf_index.faiss"
    meta_path = DATA_DIR / "pdf_meta.pkl"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(str(index_path))

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    return model, index, metadata

model, index, metadata = load_resources()

# =========================
# User Input
# =========================
if query_mode == "General MPR Issue":
    query = st.text_area(
        "Enter MPR Issue",
        height=120,
        placeholder="Describe the issue..."
    )
else:
    user_id = st.text_input(
        "Enter CaseID or User Name",
        placeholder="e.g. Kumar Sanu"
    )

run_clicked = st.button("Run")

# =========================
# Session defaults
# =========================
st.session_state.setdefault("user_summary", None)
st.session_state.setdefault("active_owner", None)

# =========================
# General MPR Flow
# =========================
if run_clicked and query_mode == "General MPR Issue":

    if not query.strip():
        st.warning("Please enter an MPR issue.")
    else:
        from services.retriever import retrieve_context
        from services.agent import pdf_agent

        # -------- RAG Recommended Solution --------
        with st.spinner("Generating recommended solution..."):
           from services.retriever import retrieve_context, format_context
           results = retrieve_context(query)
           context = format_context(results)


            if context and context.strip():
                recommended_solution = pdf_agent(query)
            else:
                recommended_solution = "No clear resolution found in historical cases."

        st.markdown("### ✅ Recommended Solution")
        st.write(recommended_solution)

        # -------- Similar Historical Cases --------
        with st.spinner("Searching similar past MPRs..."):
            results = find_similar_cases(
                query=query,
                model=model,
                index=index,
                metadata=metadata
            )

        results = sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)
        best_conf = round(results[0].get("confidence", 0), 2) if results else 0

        # =========================
        # Recommended Solution
        # =========================
        st.subheader("✅ Recommended Solution")
        recommendation_text = build_recommendation(results)

        st.markdown(
            f"""
            <div class="recommendation-card">
                <div class="recommendation-title">
                    <span>✅</span>
                    Recommended Solution
                    <span style="font-size:12px; color:#777;">
                        (derived from {best_conf}% similar historical cases)
                    </span>
                </div>
                <div class="recommendation-text">
                    {recommendation_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # =========================
        # Similar Historical Cases
        # =========================
        st.subheader("🔍 Similar Historical Cases")

        for i, r in enumerate(results, 1):
            confidence = round(r.get("confidence", 0), 2)
            label = "🟢 Best Match" if i == 1 else ""

            with st.expander(f"Case {i} {label} — Match Confidence: {confidence}%"):
                for k, v in r.items():
                    if k == "confidence" or not v:
                        continue

                    if k.lower() in ["resolution", "solution", "answer"]:
                        st.markdown(
                            f"""
                            <div style="
                                background:#f1f8f4;
                                padding:12px;
                                border-left:4px solid #2e7d32;
                                margin:10px 0;
                            ">
                                <b>✅ Solution</b><br>
                                {v}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.write(f"**{k}**: {v}")


# User / Case Insights
# =========================
if run_clicked and query_mode == "User-Specific View":
    if not user_id.strip():
        st.warning("Please enter a caseID or full name.")
    else:
        insights = get_user_or_case_insights(user_id)

        if insights["data"] is None:
            st.error("No data found for the given input.")
        else:
            if insights["type"] == "case":
                case = insights["data"]
                st.subheader(f"📄 Case Details — {case['caseid']}")
                st.json(case)
            else:
                st.session_state.user_summary = insights["data"]
                st.session_state.active_owner = insights["data"]["owner"]

# =========================
# User Summary View
# =========================
if st.session_state.user_summary:
    summary = st.session_state.user_summary
    owner = st.session_state.active_owner

    st.subheader(f"👤 User Summary — {owner}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cases", summary["total_cases"])
    c2.metric("Pending", summary["pending_cases"])
    c3.metric("Overdue (>7d)", summary["overdue_cases"])
    c4.metric("Critical (>21d)", summary["critical_cases"])

    st.json(summary["status_breakdown"])
