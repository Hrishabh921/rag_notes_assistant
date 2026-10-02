import html

import streamlit as st
from dotenv import load_dotenv
from google.genai import errors

from rag_utils import (
    get_client,
    get_page_count,
    extract_text_from_pdf,
    chunk_text,
    embed_chunks,
    embed_query,
    build_index,
    retrieve_top_chunks,
    generate_answer,
    TOP_K,
    MAX_CHUNKS,
    EMBED_MODEL,
    GEN_MODEL,
)

load_dotenv()

st.set_page_config(
    page_title="Chat with your notes",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="auto",
)

# ---------------------------------------------------------------------------
# Look and feel. Colours live in .streamlit/config.toml; this adds the parts a
# theme can't do: reading typography (Literata) and highlighter-marked sources.
# ---------------------------------------------------------------------------
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Literata:wght@400;500;600&display=swap');

:root {
  --ink: #1C2321;
  --muted: #5B6661;
  --rule: #D5DAD3;
  --sage: #EFF1EC;
  --green: #1F5F4A;
  --mark: rgba(255, 228, 92, .5);
  --serif: 'Literata', Georgia, 'Times New Roman', serif;
}

[data-testid="stMainBlockContainer"], .block-container {
  max-width: 760px; padding-top: 3.2rem; padding-bottom: 8rem;
}
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
[data-testid="stSidebar"] { border-right: 1px solid var(--rule); }

.title {
  font-family: var(--serif); font-weight: 600; font-size: 2.5rem; line-height: 1.12;
  letter-spacing: -0.015em; color: var(--ink); margin: 0 0 .7rem;
}
.lede { font-size: 1.08rem; line-height: 1.6; color: var(--muted); max-width: 58ch; margin: 0 0 2rem; text-wrap: pretty; }

.side-label { font-size: .85rem; font-weight: 600; margin: .1rem 0 .45rem; }
.side-note { font-size: .85rem; line-height: 1.45; color: var(--muted); margin: .45rem 0 .7rem; }

.doc { border-bottom: 1px solid var(--rule); padding: 0 0 1rem; margin: 0 0 1.4rem; }
.doc .name { font-weight: 600; font-size: 1.02rem; overflow-wrap: anywhere; }
.doc .meta { color: var(--muted); font-size: .9rem; margin-top: .15rem; }

.hint { color: var(--muted); font-size: .95rem; margin: 0 0 1rem; }

/* answers read like a textbook; controls stay sans-serif */
[data-testid="stChatMessageAvatarUser"] { background-color: var(--muted) !important; }
[data-testid="stChatMessageAvatarAssistant"] { background-color: var(--green) !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: var(--sage); }
[class*="st-key-answer"] p, [class*="st-key-answer"] li {
  font-family: var(--serif); font-size: 1.04rem; line-height: 1.75;
}

/* the signature element: sources marked like a highlighter on paper */
.src { margin: 0 0 1.1rem; }
.src-label { font-size: .82rem; font-weight: 600; color: var(--muted); margin-bottom: .2rem; }
.src p { font-family: var(--serif); font-size: .95rem; line-height: 1.85; margin: 0; }
.src mark {
  background: var(--mark); color: inherit; padding: .1em .04em; border-radius: 2px;
  -webkit-box-decoration-break: clone; box-decoration-break: clone;
}

.example { border-top: 1px solid var(--rule); padding-top: 1.3rem; }
.example .tag { font-size: .82rem; font-weight: 600; color: var(--muted); margin-bottom: .8rem; }
.example .q { font-weight: 600; margin: 0 0 .5rem; }
.example .a { font-family: var(--serif); font-size: 1.04rem; line-height: 1.75; margin: 0 0 1.1rem; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

EXAMPLE = (
    '<div class="example">'
    '<div class="tag">Example</div>'
    '<p class="q">What is a cache miss?</p>'
    '<p class="a">A cache miss happens when the processor asks for data that is not in the cache, '
    "so it has to be fetched from slower main memory.</p>"
    '<div class="src"><div class="src-label">Passage from your document</div>'
    "<p><mark>When the requested word is not found in the cache, a miss occurs and the whole block "
    "containing it is read from main memory into the cache.</mark></p></div>"
    "</div>"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def esc(text: str) -> str:
    """Collapse PDF line-wrap noise and escape the text for safe HTML."""
    return html.escape(" ".join(text.split()))


def api_error_message(e: errors.APIError) -> str:
    if e.code == 429:
        return (
            "Gemini's free-tier limit was reached (too many requests or tokens). Wait a minute and "
            "try again, or index fewer pages. If it keeps failing, today's quota may be used up; "
            "it resets at midnight Pacific time."
        )
    return f"Gemini API error {e.code}: {e.message}"


def passages_html(chunks: list[str]) -> str:
    return "".join(
        f'<div class="src"><div class="src-label">Passage {i}</div><p><mark>{esc(c)}</mark></p></div>'
        for i, c in enumerate(chunks, start=1)
    )


def render_message(i: int, m: dict):
    with st.chat_message(m["role"]):
        if m.get("error"):
            st.error(m["content"])
        elif m["role"] == "user":
            st.markdown(m["content"])
        else:
            with st.container(key=f"answer_{i}"):
                st.markdown(m["content"])
            if m.get("sources"):
                with st.expander(f"Sources ({len(m['sources'])} passages)"):
                    st.markdown(passages_html(m["sources"]), unsafe_allow_html=True)


def answer_question(question: str, top_k: int) -> dict:
    """Run the RAG pipeline for one question and return a chat message dict."""
    try:
        client = get_client()
        query_vector = embed_query(client, question)
        top_chunks = retrieve_top_chunks(
            st.session_state.index, query_vector, st.session_state.chunks, k=top_k
        )
        answer = generate_answer(client, question, top_chunks)
        return {"role": "assistant", "content": answer, "sources": top_chunks}
    except ValueError as e:
        return {"role": "assistant", "content": str(e), "error": True}
    except errors.APIError as e:
        return {"role": "assistant", "content": api_error_message(e), "error": True}


def index_document(uploaded_file, start_page: int, end_page: int):
    """Read the chosen pages, embed them in batches, and build the FAISS index."""
    with st.status("Indexing your document...", expanded=True) as status:
        try:
            client = get_client()
            st.write(f"Reading pages {start_page}–{end_page}...")
            text = extract_text_from_pdf(uploaded_file, start_page, end_page)
            if not text.strip():
                status.update(label="No text found on these pages", state="error")
                st.error("These pages have no selectable text (they may be scanned images). Try another range.")
                return

            chunks = chunk_text(text)
            if len(chunks) > MAX_CHUNKS:
                status.update(label="Too many pages selected", state="error")
                st.error(
                    f"Pages {start_page}–{end_page} make {len(chunks)} passages. The limit is {MAX_CHUNKS} "
                    "to stay within the free Gemini quota. Choose fewer pages."
                )
                return

            st.write(f"Embedding {len(chunks)} passages...")
            bar = st.progress(0.0)

            def show_progress(done, total, message):
                bar.progress(done / total, text=message or f"{done} of {total} passages embedded")

            vectors = embed_chunks(client, chunks, progress_callback=show_progress)

            st.write("Building the search index...")
            st.session_state.index = build_index(vectors)
            st.session_state.chunks = chunks
            st.session_state.doc = {
                "name": uploaded_file.name, "start": start_page, "end": end_page, "n": len(chunks),
            }
            st.session_state.messages = []
            status.update(label=f"Indexed {len(chunks)} passages", state="complete", expanded=False)
        except ValueError as e:
            status.update(label="Indexing failed", state="error")
            st.error(str(e))
        except errors.APIError as e:
            status.update(label="Indexing failed", state="error")
            st.error(api_error_message(e))


for key, default in {"index": None, "chunks": None, "doc": None, "messages": []}.items():
    st.session_state.setdefault(key, default)

# ---------------------------------------------------------------------------
# Sidebar: everything about the document
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="side-label">Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("PDF file", type="pdf", label_visibility="collapsed")

    total_pages = None
    start_page = end_page = 1
    if uploaded_file is not None:
        try:
            total_pages = get_page_count(uploaded_file)
        except Exception:
            st.error("Couldn't read this PDF. It may be corrupted or password-protected.")
        else:
            st.markdown(
                f'<div class="side-note">{total_pages} pages. Choose the range to index '
                "(PDF page numbers, not the numbers printed on the pages).</div>",
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns(2)
            start_page = col1.number_input("From page", 1, total_pages, 1)
            end_page = col2.number_input("To page", 1, total_pages, min(total_pages, 30))

    index_clicked = st.button("Index document", type="primary", width="stretch", disabled=not total_pages, key="index_btn")

    st.divider()
    with st.expander("Settings"):
        top_k = st.slider(
            "Passages per answer", 1, 8, TOP_K,
            help="How many of the closest passages (Top-K) are sent to Gemini as context.",
        )
    if st.button("Clear conversation", width="stretch", key="clear_btn"):
        st.session_state.messages = []
    with st.expander("How it works"):
        st.markdown(
            f"Your pages are split into short passages and turned into embeddings (`{EMBED_MODEL}`). "
            f"When you ask a question, the closest passages are found with a FAISS index, and "
            f"`{GEN_MODEL}` answers using only those passages."
        )
    if st.get_option("theme.primaryColor") != "#1F5F4A":
        st.caption("Theme not loaded: add .streamlit/config.toml (see README).")

# ---------------------------------------------------------------------------
# Main column
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="title" role="heading" aria-level="1">Chat with your notes</div>'
    '<p class="lede">Add a PDF in the sidebar, choose the pages you are studying, and ask questions. '
    "Each answer comes with the passages it was based on.</p>",
    unsafe_allow_html=True,
)

if index_clicked:
    if start_page > end_page:
        st.error("'From page' must be less than or equal to 'To page'.")
    else:
        index_document(uploaded_file, start_page, end_page)

ready = st.session_state.index is not None

if ready:
    doc = st.session_state.doc
    st.markdown(
        f'<div class="doc"><div class="name">{html.escape(doc["name"])}</div>'
        f'<div class="meta">Pages {doc["start"]}–{doc["end"]}, indexed as {doc["n"]} passages.</div></div>',
        unsafe_allow_html=True,
    )
    hint_slot = st.empty()
    if not st.session_state.messages:
        hint_slot.markdown(
            '<p class="hint">Ask a specific question. Answers come from the closest few passages, '
            "not from the whole PDF.</p>",
            unsafe_allow_html=True,
        )
else:
    st.markdown(EXAMPLE, unsafe_allow_html=True)

for i, message in enumerate(st.session_state.messages):
    render_message(i, message)

prompt = st.chat_input(
    "Ask about the pages you indexed" if ready else "Index a document to start asking",
    disabled=not ready,
)

if prompt and ready:
    hint_slot.empty()
    n = len(st.session_state.messages)
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    render_message(n, user_message)

    with st.spinner("Searching your pages..."):
        reply = answer_question(prompt, top_k)
    st.session_state.messages.append(reply)
    render_message(n + 1, reply)