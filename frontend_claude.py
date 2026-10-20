import html
import re
import time

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from database_rag_backend import (
    chatbot,
    delete_thread_checkpoints,
    ingest_pdf,
    ingest_url,
    retrieve_all_threads,
)
from database import (
    create_chat,
    ensure_chat,
    get_chats,
    update_chat_title,
    touch_chat,
    delete_chat_metadata,
)

st.set_page_config(
    page_title="NEURAL//CORE",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------
# Futuristic theme (CSS)
# ---------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=JetBrains+Mono:wght@300;400;500;700&display=swap');

:root {
    --bg-0: #03060d;
    --bg-1: #070d1a;
    --bg-2: #0b1426;
    --line: rgba(0, 240, 255, 0.22);
    --cyan: #00f0ff;
    --magenta: #ff2bd6;
    --lime: #7cff6b;
    --text: #cfe8ff;
    --muted: #6c86a8;
}

/* ---------- base ---------- */
html, body, .stApp, [class*="css"] {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    color: var(--text);
}
.stApp {
    background:
        radial-gradient(1200px 600px at 85% -10%, rgba(255, 43, 214, 0.10), transparent 60%),
        radial-gradient(900px 500px at -10% 10%, rgba(0, 240, 255, 0.10), transparent 60%),
        linear-gradient(rgba(0, 240, 255, 0.035) 1px, transparent 1px) 0 0 / 100% 36px,
        linear-gradient(90deg, rgba(0, 240, 255, 0.035) 1px, transparent 1px) 0 0 / 36px 100%,
        var(--bg-0);
    background-attachment: fixed;
}
/* subtle scanlines */
.stApp::before {
    content: "";
    position: fixed; inset: 0;
    pointer-events: none;
    background: repeating-linear-gradient(
        to bottom, rgba(255,255,255,0.018) 0px, rgba(255,255,255,0.018) 1px,
        transparent 1px, transparent 3px);
    z-index: 9999;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: var(--line); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

/* ---------- HUD header ---------- */
.hud {
    position: relative;
    padding: 18px 22px;
    margin-bottom: 18px;
    border: 1px solid var(--line);
    background: linear-gradient(135deg, rgba(0,240,255,0.06), rgba(255,43,214,0.04));
    clip-path: polygon(0 0, calc(100% - 22px) 0, 100% 22px, 100% 100%, 22px 100%, 0 calc(100% - 22px));
    box-shadow: 0 0 30px rgba(0, 240, 255, 0.08) inset;
}
.hud::after {
    content: "";
    position: absolute; left: 0; top: 0; height: 2px; width: 100%;
    background: linear-gradient(90deg, var(--cyan), var(--magenta), transparent);
    animation: sweep 4s linear infinite;
}
@keyframes sweep { 0% { opacity: .2 } 50% { opacity: 1 } 100% { opacity: .2 } }
.hud .eyebrow {
    font-size: 11px; letter-spacing: .35em; color: var(--cyan); opacity: .8;
}
.hud h1 {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700; font-size: 1.7rem; letter-spacing: .08em;
    margin: 4px 0 10px 0; padding: 0;
    background: linear-gradient(90deg, #fff, var(--cyan) 60%, var(--magenta));
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
    font-size: 11px; letter-spacing: .08em;
    padding: 3px 10px; border: 1px solid var(--line);
    color: var(--muted); background: rgba(0,0,0,0.35);
}
.chip b { color: var(--cyan); font-weight: 500; }
.chip.live b { color: var(--lime); }
.dot {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: var(--lime); margin-right: 6px;
    box-shadow: 0 0 8px var(--lime); animation: pulse 1.6s ease-in-out infinite;
}
@keyframes pulse { 50% { opacity: .35; transform: scale(.8); } }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--bg-1), var(--bg-0));
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] h1 {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 1.15rem; letter-spacing: .12em; color: var(--cyan);
    text-shadow: 0 0 12px rgba(0,240,255,.55);
}
[data-testid="stSidebar"] h3 {
    font-size: .72rem !important; letter-spacing: .3em;
    color: var(--muted); text-transform: uppercase;
}
[data-testid="stSidebar"] hr { border-color: var(--line); }

/* ---------- buttons ---------- */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    background: rgba(0, 240, 255, 0.04);
    color: var(--text);
    border: 1px solid var(--line);
    border-radius: 2px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px;
    letter-spacing: .04em;
    text-align: left;
    transition: all .18s ease;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    border-color: var(--cyan);
    color: #fff;
    background: rgba(0, 240, 255, 0.12);
    box-shadow: 0 0 14px rgba(0, 240, 255, .35), inset 0 0 10px rgba(0,240,255,.12);
}
.stButton > button:active { transform: translateY(1px); }

/* chat list cards */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--line) !important;
    border-left: 3px solid var(--cyan) !important;
    border-radius: 2px !important;
    background: rgba(7, 13, 26, 0.7);
    padding: 2px;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-left-color: var(--magenta) !important;
}

/* ---------- inputs ---------- */
.stTextInput input, .stTextArea textarea {
    background: rgba(0,0,0,.45) !important;
    border: 1px solid var(--line) !important;
    border-radius: 2px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 12px rgba(0,240,255,.35) !important;
}

/* ---------- chat ---------- */
[data-testid="stChatMessage"] {
    position: relative;
    background: linear-gradient(135deg, rgba(11,20,38,.92), rgba(7,13,26,.92));
    border: 1px solid var(--line);
    border-left: 3px solid var(--cyan);
    border-radius: 2px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: 0 0 22px rgba(0,240,255,.05);
    animation: slidein .35s ease both;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-left-color: var(--magenta);
    background: linear-gradient(135deg, rgba(30,10,36,.85), rgba(7,13,26,.92));
}
@keyframes slidein {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: none; }
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
    font-size: 14px; line-height: 1.7;
}
[data-testid="stChatMessage"] code {
    background: rgba(0,240,255,.08); color: var(--cyan);
    border: 1px solid var(--line); border-radius: 2px;
}
[data-testid="stChatMessage"] pre {
    border: 1px solid var(--line); border-radius: 2px;
    background: #02050a !important;
}

/* chat input dock */
[data-testid="stChatInput"] {
    background: rgba(3, 6, 13, .9);
    border: 1px solid var(--line);
    border-radius: 2px;
    box-shadow: 0 0 22px rgba(0,240,255,.12);
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--cyan);
    box-shadow: 0 0 26px rgba(0,240,255,.4);
}
[data-testid="stChatInput"] textarea {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text) !important;
}
[data-testid="stBottom"] > div { background: transparent; }

/* alerts / popover */
[data-testid="stAlert"] {
    border-radius: 2px; border: 1px solid var(--line);
    background: rgba(0,0,0,.45);
}
[data-testid="stPopoverBody"] {
    background: var(--bg-1); border: 1px solid var(--cyan); border-radius: 2px;
}
[data-testid="stDialog"] > div {
    border: 1px solid var(--cyan);
    background: linear-gradient(145deg, rgba(11,20,38,.98), rgba(3,6,13,.98));
    box-shadow: 0 0 36px rgba(0,240,255,.16), inset 0 0 22px rgba(0,240,255,.035);
}
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] {
    color: var(--text);
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ---------------------------
# Session state
# ---------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "last_latency" not in st.session_state:
    st.session_state.last_latency = None
 
if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

USER_AVATAR = "🧑‍💻"
AI_AVATAR = "🧠"


def message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


def load_conversation(thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)
    messages = state.values.get("messages", [])

    history = []
    for message in messages:
        if isinstance(message, HumanMessage):
            history.append({
                "role": "user",
                "content": message.content,
            })
        elif isinstance(message, AIMessage):
            history.append({
                "role": "assistant",
                "content": message.content,
            })

    return history


def open_chat(thread_id):
    st.session_state.thread_id = thread_id
    st.session_state.message_history = load_conversation(thread_id)
    st.session_state.last_latency = None


def start_new_chat():
    thread_id = create_chat()
    st.session_state.thread_id = thread_id
    st.session_state.message_history = []
    st.session_state.last_latency = None


def restore_checkpoint_chats():
    """Add older checkpoint threads to the chat list if their metadata is missing."""
    known_thread_ids = {chat["thread_id"] for chat in get_chats()}

    for thread_id in retrieve_all_threads():
        if thread_id in known_thread_ids:
            continue

        state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        first_user_message = next(
            (message for message in messages if isinstance(message, HumanMessage)),
            None,
        )
        if first_user_message is None:
            continue

        content = first_user_message.content
        title = " ".join(content.split())[:60] if isinstance(content, str) else ""
        ensure_chat(
            thread_id,
            title or f"Restored chat {thread_id[:8]}",
            state.created_at,
        )
        known_thread_ids.add(thread_id)


@st.dialog("Rename session", icon=":material/edit:")
def rename_chat_dialog(thread_id, current_title):
    st.caption("SESSION MANAGEMENT / EDIT TITLE")
    st.write("Choose a clear name for this conversation.")

    with st.form(f"rename_chat_form_{thread_id}"):
        new_title = st.text_input(
            "Session name",
            value=current_title,
            max_chars=100,
            key=f"rename_title_{thread_id}",
        )
        save_col, cancel_col = st.columns(2)
        with save_col:
            save = st.form_submit_button(
                "Save changes",
                type="primary",
                width="stretch",
            )
        with cancel_col:
            cancel = st.form_submit_button(
                "Cancel",
                width="stretch",
            )

    if cancel:
        st.rerun()

    if save:
        if new_title.strip():
            update_chat_title(thread_id, new_title.strip())
            st.rerun()
        else:
            st.error("Session name cannot be empty.")


@st.dialog("Delete session?", icon=":material/delete_forever:")
def delete_chat_dialog(thread_id, title):
    st.caption("SESSION MANAGEMENT / PERMANENT ACTION")
    st.write("This will permanently delete the conversation:")
    st.text(title)
    st.warning("This action cannot be undone.")

    confirm_col, cancel_col = st.columns(2)
    with confirm_col:
        confirm = st.button(
            "Delete session",
            key=f"confirm_delete_{thread_id}",
            type="primary",
            width="stretch",
        )
    with cancel_col:
        cancel = st.button(
            "Cancel",
            key=f"cancel_delete_{thread_id}",
            width="stretch",
        )

    if cancel:
        st.rerun()

    if confirm:
        try:
            delete_thread_checkpoints(thread_id)
            delete_chat_metadata(thread_id)
        except Exception as exc:
            st.error(f"Delete failed: {exc}")
        else:
            if st.session_state.thread_id == thread_id:
                st.session_state.thread_id = None
                st.session_state.message_history = []
            st.rerun()


# Restore checkpoint-only conversations once per Streamlit session.
if not st.session_state.get("checkpoint_chats_restored", False):
    restore_checkpoint_chats()
    st.session_state["checkpoint_chats_restored"] = True

# Create an initial conversation on first load
if st.session_state.thread_id is None:
    chats = get_chats()

    if chats:
        open_chat(chats[0]["thread_id"])
    else:
        start_new_chat()

thread_key = str(st.session_state.thread_id)
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})


# ---------------------------
# Sidebar: conversation CRUD
# ---------------------------
with st.sidebar:
    st.title("◈ NEURAL//CORE")
    st.caption("LANGGRAPH · AGENT RUNTIME · v1.0")

    if st.button("＋  INIT NEW SESSION", width="stretch"):
        start_new_chat()
        st.rerun()

    st.divider()
    st.subheader("Session Logs")


with st.sidebar:
    if thread_docs:
        latest_doc = list(thread_docs.values())[-1]
        st.success(
            f"Using `{latest_doc.get('filename')}` "
            f"({latest_doc.get('chunks')} chunks indexed)"
        )
    else:
        st.info("No sources indexed yet.")

    uploaded_pdf = st.file_uploader("Upload a PDF for this chat", type=["pdf"])
    if uploaded_pdf:
        if uploaded_pdf.name in thread_docs:
            st.info(f"`{uploaded_pdf.name}` already processed for this chat.")
        else:
            try:
                with st.status("Indexing PDF…", expanded=True) as status_box:
                    summary = ingest_pdf(
                        uploaded_pdf.getvalue(),
                        thread_id=thread_key,
                        filename=uploaded_pdf.name,
                    )
                    thread_docs[uploaded_pdf.name] = summary
                    status_box.update(
                        label="PDF indexed", state="complete", expanded=False
                    )
            except Exception as exc:
                st.error(f"PDF indexing failed: {exc}")

    with st.form(f"web_page_form_{thread_key}"):
        page_url = st.text_input(
            "Add a web page URL",
            placeholder="https://example.com/article",
        )
        index_page = st.form_submit_button("Index web page", width="stretch")

    if index_page:
        if page_url in thread_docs:
            st.info("This web page is already indexed for this chat.")
        else:
            try:
                with st.status("Fetching and indexing web page...", expanded=True) as status_box:
                    summary = ingest_url(page_url, thread_id=thread_key)
                    thread_docs[summary["filename"]] = summary
                    status_box.update(
                        label="Web page indexed",
                        state="complete",
                        expanded=False,
                    )
            except Exception as exc:
                st.error(f"Web page indexing failed: {exc}")

    st.divider()
    search = st.text_input(
        "Search conversations",
        placeholder="> query logs...",
        label_visibility="collapsed",
    )

    for chat in get_chats():
        thread_id = chat["thread_id"]
        title = chat["title"]

        if search.lower() not in title.lower():
            continue

        is_active = thread_id == st.session_state.thread_id
        label = f"{'▶ ' if is_active else '  '}{title}"

        with st.container(border=True):
            if st.button(
                label,
                key=f"open_{thread_id}",
                width="stretch",
            ):
                open_chat(thread_id)
                st.rerun()

            with st.popover("⋯"):
                if st.button(
                    "Rename",
                    key=f"rename_{thread_id}",
                    width="stretch",
                ):
                    rename_chat_dialog(thread_id, title)

                if st.button(
                    "Delete",
                    key=f"delete_{thread_id}",
                    width="stretch",
                ):
                    delete_chat_dialog(thread_id, title)


# ---------------------------
# Main chat window
# ---------------------------
chats = get_chats()
active_chat = next(
    (
        chat for chat in chats
        if chat["thread_id"] == st.session_state.thread_id
    ),
    None,
)

active_title = active_chat["title"] if active_chat else "New Chat"
msg_count = len(st.session_state.message_history)
short_id = html.escape(str(st.session_state.thread_id)[:8])
latency = st.session_state.last_latency
latency_txt = f"{latency:.2f}s" if latency is not None else "--"

st.markdown(
    f"""
    <div class="hud">
        <div class="eyebrow">// ACTIVE SESSION</div>
        <h1>{html.escape(active_title)}</h1>
        <div class="chips">
            <span class="chip live"><span class="dot"></span><b>ONLINE</b></span>
            <span class="chip">THREAD <b>{short_id}</b></span>
            <span class="chip">MSGS <b>{msg_count}</b></span>
            <span class="chip">LAST RESPONSE <b>{latency_txt}</b></span>
            <span class="chip">STREAM <b>TOKEN</b></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render existing messages
for message in st.session_state.message_history:
    avatar = USER_AVATAR if message["role"] == "user" else AI_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


URL_PATTERN = re.compile(r"https?://[^\s<>\"')\]]+")


def extract_urls(text):
    """Return http(s) URLs found in a chat message."""
    return [match.group(0).rstrip(".,;:!?") for match in URL_PATTERN.finditer(text)]


def index_links_in_message(text, thread_key, thread_docs):
    """Index any web page URLs the user pasted directly into the chat box."""
    indexed = []
    for url in extract_urls(text):
        if url in thread_docs:
            continue
        try:
            summary = ingest_url(url, thread_id=thread_key)
            thread_docs[summary["filename"]] = summary
            indexed.append(summary)
        except Exception as exc:
            st.warning(f"Could not index {url}: {exc}")
    return indexed


# ---------------------------
# Chat input + streaming
# ---------------------------
user_input = st.chat_input("> enter prompt...")

if user_input:
    thread_id = st.session_state.thread_id

    # Index any links pasted directly into the chat before answering.
    index_links_in_message(user_input, thread_key, thread_docs)

    # Only use the first question to create the initial title.
    is_first_question = not any(
        message["role"] == "user"
        for message in st.session_state.message_history
    )

    if is_first_question:
        title = " ".join(user_input.split())[:60]
        update_chat_title(thread_id, title)

    st.session_state.message_history.append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_input)

    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }

    started = time.perf_counter()

    try:
        with st.chat_message("assistant", avatar=AI_AVATAR):
            with st.spinner("Thinking..."):
                state = chatbot.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                )
            response = next(
                (
                    message_text(message.content).strip()
                    for message in reversed(state.get("messages", []))
                    if isinstance(message, AIMessage)
                    and message_text(message.content).strip()
                ),
                "",
            )
            if not response:
                raise RuntimeError("The assistant completed without a text reply.")
            st.markdown(response)
    except Exception as exc:
        st.error(f"Assistant request failed: {exc}")
    else:
        st.session_state.last_latency = time.perf_counter() - started

        st.session_state.message_history.append({
            "role": "assistant",
            "content": response,
        })

        touch_chat(thread_id)
        st.rerun()