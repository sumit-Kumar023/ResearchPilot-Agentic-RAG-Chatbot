import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from backend.btw_handler import handle_btw
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.rag_graph import build_graph
from backend.vector_store import add_paper, list_papers
import sqlite3

st.set_page_config(
    page_title="ResearchPilot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS Injection ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #0F1117 !important;
    color: #F0EEF8 !important;
    font-size: 16px !important;
}

/* Force all text to be legible by default */
p, span, div, li, td, th, label {
    color: #F0EEF8 !important;
    font-size: 15px !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #13151F !important;
    border-right: 1px solid #1E2133 !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
[data-testid="stSidebarContent"] {
    padding: 0 !important;
}

/* ── Sidebar Header Brand ── */
.rp-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 20px 16px 16px;
    border-bottom: 1px solid #1E2133;
    margin-bottom: 4px;
}
.rp-brand-icon {
    width: 34px;
    height: 34px;
    background: linear-gradient(135deg, #7C6AF7, #5B4ED4);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    box-shadow: 0 0 12px rgba(124, 106, 247, 0.35);
    flex-shrink: 0;
}
.rp-brand-text {
    font-size: 17px;
    font-weight: 700;
    color: #F0EEF8;
    letter-spacing: -0.01em;
}
.rp-brand-sub {
    font-size: 11px;
    font-weight: 400;
    color: #9CA3AF;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Sidebar Section Labels ── */
.rp-section-label {
    font-size: 11px;
    font-weight: 700;
    color: #9CA3AF !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 12px 16px 6px;
}

/* ── New Chat Button ── */
div[data-testid="stButton"] > button[kind="primary"].new-chat-btn,
.stButton > button:has(p:contains("+ New Chat")) {
    background: linear-gradient(135deg, #7C6AF7 0%, #5B4ED4 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
    width: 100%;
    transition: opacity 0.15s ease, transform 0.1s ease !important;
    box-shadow: 0 2px 8px rgba(124, 106, 247, 0.3) !important;
}

/* ── Session buttons ── */
[data-testid="stSidebar"] .stButton > button {
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: background-color 0.15s ease !important;
    border: none !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background-color: transparent !important;
    color: #D1D5DB !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background-color: #1E2133 !important;
    color: #F0EEF8 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background-color: #1E2133 !important;
    color: #F0EEF8 !important;
    font-weight: 600 !important;
}

/* Delete button */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton:last-child > button {
    background-color: transparent !important;
    color: #4B5563 !important;
    padding: 4px 8px !important;
    font-size: 12px !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton:last-child > button:hover {
    background-color: rgba(239, 68, 68, 0.12) !important;
    color: #EF4444 !important;
}

/* ── Document section in sidebar ── */
[data-testid="stSidebar"] .stFileUploader {
    background-color: #1A1D2E !important;
    border: 1.5px dashed #2D3148 !important;
    border-radius: 10px !important;
    padding: 12px !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stSidebar"] .stFileUploader:hover {
    border-color: #7C6AF7 !important;
}
[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stTextArea > div > div > textarea {
    background-color: #1A1D2E !important;
    border: 1px solid #2D3148 !important;
    border-radius: 8px !important;
    color: #F0EEF8 !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] .stTextInput > div > div > input:focus,
[data-testid="stSidebar"] .stTextArea > div > div > textarea:focus {
    border-color: #7C6AF7 !important;
    box-shadow: 0 0 0 2px rgba(124, 106, 247, 0.2) !important;
}

/* ── Main content area ── */
[data-testid="stMain"] {
    background-color: #0F1117 !important;
}
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Page header ── */
.rp-header {
    padding: 24px 32px 0;
}
.rp-header h1 {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #E8E6F0 !important;
    margin: 0 0 4px !important;
    letter-spacing: -0.02em;
}
.rp-header-meta {
    font-size: 13px;
    color: #6B7280;
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
}
.rp-header-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #1A1D2E;
    border: 1px solid #2D3148;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 13px;
    font-weight: 500;
    color: #D1D5DB !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessageContent"] {
    background: transparent !important;
}
[data-testid="stChatMessage"] {
    padding: 12px 32px !important;
    border-radius: 0 !important;
    background: transparent !important;
    border-bottom: 1px solid #1A1D2E !important;
    transition: background-color 0.15s ease !important;
}
[data-testid="stChatMessage"]:hover {
    background-color: rgba(26, 29, 46, 0.4) !important;
}

/* User message */
[data-testid="stChatMessage"][data-testid*="user"] {
    border-left: 3px solid #7C6AF7 !important;
}

/* AI message */
[data-testid="stChatMessage"]:not([data-testid*="user"]) {
    border-left: 3px solid transparent !important;
}

/* Avatar styling */
[data-testid="stChatMessageAvatar"] {
    border-radius: 8px !important;
    overflow: hidden;
}

/* Message text */
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] span {
    font-size: 16px !important;
    line-height: 1.75 !important;
    color: #F0EEF8 !important;
}

/* Code blocks */
[data-testid="stChatMessageContent"] pre {
    background-color: #1A1D2E !important;
    border: 1px solid #2D3148 !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
}
[data-testid="stChatMessageContent"] code {
    font-size: 12.5px !important;
    color: #A78BFA !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    border-top: 1px solid #1E2133 !important;
    padding: 16px 32px !important;
    background-color: #0F1117 !important;
    position: sticky !important;
    bottom: 0 !important;
}
[data-testid="stChatInput"] > div {
    background-color: #1A1D2E !important;
    border: 1.5px solid #2D3148 !important;
    border-radius: 12px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stChatInput"]:focus-within > div {
    border-color: #7C6AF7 !important;
    box-shadow: 0 0 0 3px rgba(124, 106, 247, 0.15) !important;
}
[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: #F0EEF8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #6B7280 !important;
}

/* ── Expander (graph state) ── */
[data-testid="stExpander"] {
    background-color: #13151F !important;
    border: 1px solid #1E2133 !important;
    border-radius: 10px !important;
    margin-top: 10px !important;
}
[data-testid="stExpander"] summary {
    font-size: 13px !important;
    color: #9CA3AF !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
}
[data-testid="stExpander"] summary:hover {
    color: #F0EEF8 !important;
}

/* ── Toast / Alert ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 12px !important;
    padding: 10px 14px !important;
}
.stSuccess {
    background-color: rgba(74, 222, 128, 0.08) !important;
    border: 1px solid rgba(74, 222, 128, 0.25) !important;
    color: #4ADE80 !important;
}
.stWarning {
    background-color: rgba(245, 158, 11, 0.08) !important;
    border: 1px solid rgba(245, 158, 11, 0.25) !important;
    color: #F59E0B !important;
}
.stError {
    background-color: rgba(239, 68, 68, 0.08) !important;
    border: 1px solid rgba(239, 68, 68, 0.25) !important;
    color: #EF4444 !important;
}
.stInfo {
    background-color: rgba(124, 106, 247, 0.08) !important;
    border: 1px solid rgba(124, 106, 247, 0.25) !important;
    color: #A78BFA !important;
}

/* ── Dividers ── */
hr {
    border-color: #1E2133 !important;
    margin: 10px 0 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #7C6AF7 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2D3148; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #3D4168; }

/* ── Doc list ── */
.rp-doc-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    background: #1A1D2E;
    border: 1px solid #1E2133;
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 13px;
    color: #D1D5DB !important;
    transition: border-color 0.15s ease;
}
.rp-doc-item:hover { border-color: #7C6AF7; color: #F0EEF8 !important; }
.rp-doc-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #4ADE80;
    flex-shrink: 0;
    box-shadow: 0 0 6px rgba(74, 222, 128, 0.5);
}

/* ── Empty state ── */
.rp-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 32px;
    text-align: center;
    gap: 12px;
}
.rp-empty-icon {
    font-size: 40px;
    opacity: 0.5;
}
.rp-empty-title {
    font-size: 20px;
    font-weight: 600;
    color: #D1D5DB !important;
}
.rp-empty-sub {
    font-size: 15px;
    color: #9CA3AF !important;
    line-height: 1.6;
    max-width: 380px;
}

/* ── Capability pills ── */
.rp-caps {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 8px;
}
.rp-cap-pill {
    background: #1A1D2E;
    border: 1px solid #2D3148;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 500;
    color: #D1D5DB !important;
}

/* ── /btw badge ── */
.rp-btw-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.25);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 10px;
    font-weight: 600;
    color: #F59E0B;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* ── Confirm delete card ── */
.rp-confirm {
    background: rgba(239, 68, 68, 0.06);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 8px;
    padding: 8px 10px;
    margin: 4px 0;
    font-size: 12px;
    color: #FCA5A5;
}

/* Sidebar markdown text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: #D1D5DB !important;
    font-size: 14px !important;
}
[data-testid="stSidebar"] strong {
    color: #F0EEF8 !important;
    font-size: 14px !important;
}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: #9CA3AF !important;
    font-size: 12px !important;
}

/* Button sizing fix in sidebar */
[data-testid="stSidebar"] .stButton > button {
    padding: 6px 10px !important;
    min-height: 34px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers (unchanged logic) ─────────────────────────────────────────────────
def delete_langgraph_thread(session_id: str, db_path: str = "checkpoints.db"):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to delete LangGraph thread {session_id}: {e}")


@st.cache_resource
def get_graph():
    return build_graph()


SESSIONS_FILE = Path("sessions.json")
_rename_llm = ChatOpenAI(model="gpt-4o-mini")


def load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


def _serialize_state(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if k == "messages":
            out[k] = [
                {
                    "type": type(m).__name__,
                    "content": (
                        m.content[:300] if isinstance(m.content, str) else repr(m.content)[:300]
                    ),
                }
                for m in (v or [])
            ]
        elif k == "retrieved_docs":
            out[k] = [
                {"content": d.page_content[:300], "metadata": d.metadata}
                for d in (v or [])
            ]
        else:
            out[k] = v
    return out


def generate_session_name(first_message: str) -> str:
    try:
        response = _rename_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-5 word title for a research chat session "
                        "based on the user's first message. Return only the title, "
                        "no punctuation at the end, no quotes."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ]
        )
        return response.content.strip()
    except Exception:
        return "New Session"


def maybe_rename_session(session_id: str, first_message: str) -> None:
    if st.session_state.sessions_meta.get(session_id, {}).get("is_named"):
        return
    name = generate_session_name(first_message)
    st.session_state.sessions_meta[session_id]["name"] = name
    st.session_state.sessions_meta[session_id]["is_named"] = True
    save_sessions(st.session_state.sessions_meta)


def create_session() -> str:
    sid = str(uuid.uuid4())
    st.session_state.sessions_meta[sid] = {
        "id": sid,
        "name": "New Session",
        "created_at": datetime.now().isoformat(),
        "is_named": False,
    }
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats[sid] = []
    st.session_state.turns[sid] = 0
    return sid


def delete_session(session_id: str) -> None:
    delete_langgraph_thread(session_id)
    st.session_state.sessions_meta.pop(session_id, None)
    st.session_state.chats.pop(session_id, None)
    st.session_state.turns.pop(session_id, None)
    save_sessions(st.session_state.sessions_meta)
    if session_id == st.session_state.active_session_id:
        if st.session_state.sessions_meta:
            latest = max(
                st.session_state.sessions_meta.values(),
                key=lambda s: s["created_at"],
            )
            st.session_state.active_session_id = latest["id"]
        else:
            new_sid = create_session()
            st.session_state.active_session_id = new_sid


def load_session_chats(session_id: str) -> list[dict]:
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            return []
        chats = []
        turn = 0
        for msg in state.values.get("messages", []):
            type_name = type(msg).__name__
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if type_name == "HumanMessage":
                chats.append({"role": "user", "content": content})
            elif type_name in ("AIMessage", "AIMessageChunk"):
                turn += 1
                chats.append({"role": "assistant", "content": content, "turn": turn, "graph_state": {}})
        return chats
    except Exception:
        return []


def switch_session(session_id: str) -> None:
    st.session_state.active_session_id = session_id
    if session_id not in st.session_state.chats:
        st.session_state.chats[session_id] = load_session_chats(session_id)
    if session_id not in st.session_state.turns:
        turn_count = sum(1 for m in st.session_state.chats[session_id] if m["role"] == "assistant")
        st.session_state.turns[session_id] = turn_count


graph = get_graph()

# ── Bootstrap ──────────────────────────────────────────────────────────────────
if "sessions_meta" not in st.session_state:
    st.session_state.sessions_meta = load_sessions()
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "turns" not in st.session_state:
    st.session_state.turns = {}
if "active_session_id" not in st.session_state:
    if st.session_state.sessions_meta:
        latest = max(
            st.session_state.sessions_meta.values(),
            key=lambda s: s["created_at"],
        )
        switch_session(latest["id"])
    else:
        sid = create_session()
        st.session_state.active_session_id = sid

active_sid = st.session_state.active_session_id

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand header
    st.markdown("""
    <div class="rp-brand">
        <div class="rp-brand-icon">🔬</div>
        <div>
            <div class="rp-brand-text">ResearchPilot</div>
            <div class="rp-brand-sub">AI Research Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # New Chat button
    if st.button("＋  New Chat", use_container_width=True, type="primary"):
        new_sid = create_session()
        st.session_state.active_session_id = new_sid
        active_sid = new_sid
        st.rerun()

    # Sessions
    st.markdown('<div class="rp-section-label">Conversations</div>', unsafe_allow_html=True)

    sorted_sessions = sorted(
        st.session_state.sessions_meta.values(),
        key=lambda s: s["created_at"],
        reverse=True,
    )

    for session in sorted_sessions:
        sid = session["id"]
        is_active = sid == st.session_state.active_session_id
        btn_type = "primary" if is_active else "secondary"
        name = session["name"]
        # Truncate long names
        display_name = (name[:26] + "…") if len(name) > 28 else name

        col1, col2 = st.columns([6, 1])
        with col1:
            if st.button(
                f"{'💬 ' if is_active else '  '}{display_name}",
                key=f"sess_{sid}",
                use_container_width=True,
                type=btn_type,
            ):
                if not is_active:
                    switch_session(sid)
                    st.rerun()
        with col2:
            if st.button("🗑", key=f"delete_{sid}", use_container_width=True):
                st.session_state[f"confirm_delete_{sid}"] = True

        if st.session_state.get(f"confirm_delete_{sid}", False):
            st.markdown(f'<div class="rp-confirm">Delete "{name[:20]}"?</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Delete", key=f"yes_delete_{sid}", use_container_width=True):
                    delete_session(sid)
                    st.session_state.pop(f"confirm_delete_{sid}", None)
                    st.rerun()
            with c2:
                if st.button("Cancel", key=f"no_delete_{sid}", use_container_width=True):
                    st.session_state.pop(f"confirm_delete_{sid}", None)
                    st.rerun()

    st.divider()

    # Documents section
    st.markdown('<div class="rp-section-label">Knowledge Base</div>', unsafe_allow_html=True)

    # File upload
    st.markdown("**Upload Files**", unsafe_allow_html=False)
    uploaded_files = st.file_uploader(
        "Drop PDF, TXT, or Markdown",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        key=f"uploader_{active_sid}",
        label_visibility="collapsed",
    )
    if st.button("Add Files", use_container_width=True, key="btn_add_files"):
        if uploaded_files:
            processed_key = f"processed_files_{active_sid}"
            if processed_key not in st.session_state:
                st.session_state[processed_key] = set()
            with st.spinner("Indexing files…"):
                for f in uploaded_files:
                    if f.name in st.session_state[processed_key]:
                        st.info(f"Already loaded: {f.name}")
                        continue
                    suffix = Path(f.name).suffix
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(f.read())
                            tmp_path = tmp.name
                        docs = load_document(tmp_path)
                        for doc in docs:
                            doc.metadata["title"] = Path(f.name).stem
                        add_paper(docs, active_sid)
                        st.session_state[processed_key].add(f.name)
                        st.success(f"✓ {f.name}")
                    except Exception as e:
                        st.error(f"✗ {f.name}: {e}")
                    finally:
                        if tmp_path:
                            Path(tmp_path).unlink(missing_ok=True)
            st.rerun()
        else:
            st.warning("Select files first.")

    st.markdown("**Web Pages**")
    url_input = st.text_area(
        "URLs",
        key=f"url_area_{active_sid}",
        height=72,
        label_visibility="collapsed",
        placeholder="https://arxiv.org/abs/...",
    )
    if st.button("Load URLs", use_container_width=True, key="btn_load_urls"):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        if urls:
            with st.spinner("Fetching pages…"):
                for url in urls:
                    try:
                        docs = load_webpage(url)
                        add_paper(docs, active_sid)
                        st.success(f"✓ {url[:50]}")
                    except Exception as e:
                        st.error(f"✗ {url[:40]}: {e}")
            st.rerun()
        else:
            st.warning("Enter at least one URL.")

    st.markdown("**ArXiv**")
    arxiv_title = st.text_input(
        "Title or ArXiv ID",
        key=f"arxiv_input_{active_sid}",
        label_visibility="collapsed",
        placeholder="1706.03762 or Attention Is All You Need",
    )
    if st.button("Load Paper", use_container_width=True, key="btn_load_arxiv"):
        if arxiv_title.strip():
            with st.spinner("Fetching from ArXiv…"):
                try:
                    docs = load_arxiv(arxiv_title.strip())
                    add_paper(docs, active_sid)
                    loaded_title = docs[0].metadata.get("title") if docs else arxiv_title.strip()
                    st.success(f"✓ {loaded_title[:50]}")
                except Exception as e:
                    st.error(f"Failed: {e}")
            st.rerun()
        else:
            st.warning("Enter a paper title or ArXiv ID.")

    # Loaded documents list
    st.divider()
    st.markdown('<div class="rp-section-label">Indexed Documents</div>', unsafe_allow_html=True)
    try:
        doc_titles = list_papers(active_sid)
    except Exception:
        doc_titles = None

    if doc_titles is None:
        st.caption("⚠ Could not load list — try refreshing.")
    elif doc_titles:
        for title in doc_titles:
            short = (title[:32] + "…") if len(title) > 35 else title
            st.markdown(
                f'<div class="rp-doc-item"><span class="rp-doc-dot"></span>{short}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="padding: 8px 0; font-size: 12px; color: #4B5563;">No documents indexed yet.</div>',
            unsafe_allow_html=True,
        )

# ── Main area ──────────────────────────────────────────────────────────────────
# Session header
active_session_name = st.session_state.sessions_meta.get(active_sid, {}).get("name", "Session")
chat_count = len([m for m in st.session_state.chats.get(active_sid, []) if m["role"] == "user"])

try:
    doc_count = len(list_papers(active_sid) or [])
except Exception:
    doc_count = 0

st.markdown(f"""
<div class="rp-header">
    <h1>{active_session_name}</h1>
    <div class="rp-header-meta">
        <span class="rp-header-pill">💬 {chat_count} message{"s" if chat_count != 1 else ""}</span>
        <span class="rp-header-pill">📄 {doc_count} document{"s" if doc_count != 1 else ""} indexed</span>
        <span class="rp-header-pill" style="color: #A78BFA; border-color: rgba(124,106,247,0.3);">🔬 Agentic RAG</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Chat display ───────────────────────────────────────────────────────────────
messages = st.session_state.chats.get(active_sid, [])

if not messages:
    st.markdown("""
    <div class="rp-empty">
        <div class="rp-empty-icon">🔬</div>
        <div class="rp-empty-title">Start a research conversation</div>
        <div class="rp-empty-sub">
            Upload papers in the sidebar, then ask questions, verify claims,
            or explore connections across your documents.
        </div>
        <div class="rp-caps">
            <span class="rp-cap-pill">📄 Answer from papers</span>
            <span class="rp-cap-pill">✅ Verify claims</span>
            <span class="rp-cap-pill">🌐 Search the web</span>
            <span class="rp-cap-pill">🔍 /btw side queries</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                with st.expander(f"📊 Graph state · turn {msg['turn']}", expanded=False):
                    st.json(msg["graph_state"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your papers, verify a claim, or search the web…  (tip: /btw for side queries)"):
    is_btw = prompt.strip().lower().startswith("/btw")

    if is_btw:
        query = prompt.strip()[4:].strip()
        with st.chat_message("user"):
            st.markdown(
                '<div class="rp-btw-badge">⚡ Side Channel</div>',
                unsafe_allow_html=True,
            )
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.markdown(
                '<div class="rp-btw-badge">⚡ Side Channel · not saved</div>',
                unsafe_allow_html=True,
            )
            if not query:
                st.markdown("Add a question after `/btw`, e.g. `/btw What is attention?`")
            else:
                placeholder = st.empty()
                response_text = ""
                for chunk in handle_btw(query):
                    response_text += chunk
                    placeholder.markdown(response_text + "▌")
                placeholder.markdown(response_text)

    else:
        if active_sid not in st.session_state.chats:
            st.session_state.chats[active_sid] = []
        if active_sid not in st.session_state.turns:
            st.session_state.turns[active_sid] = 0

        is_first_message = len(st.session_state.chats[active_sid]) == 0

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chats[active_sid].append({"role": "user", "content": prompt})
        st.session_state.turns[active_sid] += 1
        current_turn = st.session_state.turns[active_sid]

        if is_first_message:
            maybe_rename_session(active_sid, prompt)

        input_state = {
            "messages": [HumanMessage(content=prompt)],
            "session_id": active_sid,
            "query": prompt,
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
        }
        config = {"configurable": {"thread_id": active_sid}}

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Thinking…_ ▌")
            response_text = ""

            for chunk, metadata in graph.stream(input_state, config, stream_mode="messages"):
                if (
                    metadata.get("langgraph_node") == "generate_answer"
                    and hasattr(chunk, "content")
                    and chunk.content
                ):
                    response_text += chunk.content
                    placeholder.markdown(response_text + "▌")

            if not response_text:
                final_values = graph.get_state(config).values
                response_text = final_values.get("answer") or "No response generated."

            placeholder.markdown(response_text)

            final_values = graph.get_state(config).values
            state_snapshot = _serialize_state(final_values)

            with st.expander(f"📊 Graph state · turn {current_turn}", expanded=False):
                st.json(state_snapshot)

        st.session_state.chats[active_sid].append(
            {
                "role": "assistant",
                "content": response_text,
                "graph_state": state_snapshot,
                "turn": current_turn,
            }
        )

        if is_first_message:
            st.rerun()
