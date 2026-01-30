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
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =========================
# Helper: Build Recommendation
# =========================
def build_recommendation(results, top_n=3):
    recommendations = []

    for r in results[:top_n]:
        if r.get("Resolution"):
            recommendations.append(f"• {r['Resolution']}")
        elif r.get("details"):
            recommendations.append(f"• {r['details']}")

    if not recommendations:
        return "No clear resolution found in historical cases."

    return "\n".join(recommendations)

# =========================
# Centered Header (Logo + Query Type)
# =========================
scale = "2k"  # fixed scale

with st.container():

    _, logo_col, _ = st.columns([1, 1, 1])
    with logo_col:
        st.image("assets/logo.png", width=280)

    st.markdown("<div style='margin-top:-20px'></div>", unsafe_allow_html=True)

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
# Load heavy resources
# =========================
@st.cache_resource
def load_resources(scale):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(f"data/case_index_{scale}.faiss")
    with open(f"data/case_meta_{scale}.pkl", "rb") as f:
        metadata = pickle.load(f)
    return model, index, metadata

model, index, metadata = load_resources(scale)

# =========================
# User Input Section
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
# Session State Init
# =========================
if "user_summary" not in st.session_state:
    st.session_state.user_summary = None

if "active_owner" not in st.session_state:
    st.session_state.active_owner = None

# =========================
# General MPR Flow
# =========================
if run_clicked and query_mode == "General MPR Issue":
    if not query.strip():
        st.warning("Please enter an MPR issue.")
    else:
        with st.spinner("Searching similar past MPRs..."):
            results = find_similar_cases(
                query=query,
                model=model,
                index=index,
                metadata=metadata
            )

        results = sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)

        # =========================
        # ✅ Recommended Solution
        # =========================
        st.subheader("✅ Recommended Solution")
        recommendation_text = build_recommendation(results)
        st.markdown(
    f"""
    <div class="recommendation-card">
        <div class="recommendation-title">
            <span>✅</span>
            Recommended Solution
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

        IMPORTANT_FIELDS = [
            "caseid",
            "category",
            "statuscode",
            "currentowner",
            "reportedon",
            "aging",
            "subject",
            "details",
            "Resolution"
        ]

        for i, r in enumerate(results, 1):
            confidence = round(r.get("confidence", 0), 2)
            label = "🟢 Best Match" if i == 1 else ""

            with st.expander(f"Case {i} {label} — Match Confidence: {confidence}%"):
                for field in IMPORTANT_FIELDS:
                    if field in r and r[field]:
                        st.write(f"**{field}**: {r[field]}")

# =========================
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
                st.session_state.user_summary = None
                st.session_state.active_owner = None
            else:
                summary = insights["data"]
                st.session_state.user_summary = summary
                st.session_state.active_owner = summary["owner"]

# =========================
# User Summary View
# =========================
if st.session_state.user_summary is not None:
    summary = st.session_state.user_summary
    owner = st.session_state.active_owner

    st.subheader(f"👤 User Summary — {owner}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cases", summary["total_cases"])
    c2.metric("Pending", summary["pending_cases"])
    c3.metric("Overdue (>7d)", summary["overdue_cases"])
    c4.metric("Critical (>21d)", summary["critical_cases"])

    st.markdown("### Status Breakdown")
    st.json(summary["status_breakdown"])

    st.markdown("---")
    st.subheader("📌 Focused Case View")

    case_type = st.radio(
        "Select case category",
        ["Pending", "Overdue", "Critical"],
        horizontal=True
    )

    if case_type == "Pending":
        cases = get_pending_cases(owner)
        badge = "🟡 Pending"
        empty_msg = "No pending cases found for this user."
    elif case_type == "Overdue":
        cases = get_overdue_cases(owner)
        badge = "🟠 Overdue"
        empty_msg = "No overdue cases found for this user."
    else:
        cases = get_critical_cases(owner)
        badge = "🔴 Critical"
        empty_msg = "No critical cases found for this user."

    if not cases:
        st.info(empty_msg)
    else:
        for c in cases:
            with st.expander(f"{badge} | Case {c['caseid']} | Aging: {c['aging']} days"):
                st.write(f"**Category:** {c['category']}")
                st.write(f"**Status:** {c['statuscode']}")
                st.write(f"**Reported On:** {c['reportedon']}")
                st.write(f"**Subject:** {c.get('subject','')}")
                st.write(f"**Details:** {c.get('details','')}")
