import os
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from pypdf import PdfReader

from dedupe import unique_search_results


BASE_DIR = Path(__file__).resolve().parent
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "German_Basics"
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"

load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="Grammatik · German Tutor",
    page_icon="🇩🇪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    :root {
        --ink: #17243b;
        --muted: #718096;
        --paper: #f7f8fb;
        --blue: #3459d1;
        --line: #e8ebf2;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(247,248,251,.86); }
    [data-testid="stSidebar"] {
        background: #fff;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.6rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {
        color: #53627b !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] strong { color: var(--ink) !important; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .block-container { max-width: 1160px; padding-top: 2rem; padding-bottom: 5rem; }
    .brand { display:flex; align-items:center; gap:12px; margin-bottom:2.1rem; }
    .brand-mark {
        display:flex; align-items:center; justify-content:center; width:44px; height:44px;
        border-radius:15px; color:#fff; font-size:22px;
        background:linear-gradient(145deg,#5678f4,#2948b7);
        box-shadow:0 8px 20px rgba(52,89,209,.25);
    }
    .brand-name { font-family:'Playfair Display',serif; font-size:21px; font-weight:700; color:var(--ink); }
    .brand-caption { color:var(--muted); font-size:11px; letter-spacing:.12em; text-transform:uppercase; }
    .hero {
        position:relative; overflow:hidden; padding:34px 38px; margin:4px 0 25px;
        color:white; border-radius:24px;
        background:linear-gradient(115deg,#18294c 0%,#283e78 58%,#405dc0 100%);
        box-shadow:0 18px 45px rgba(30,48,93,.14);
    }
    .hero:after {
        content:''; position:absolute; width:280px; height:280px; right:-70px; top:-145px;
        border:1px solid rgba(255,255,255,.16); border-radius:50%;
        box-shadow:0 0 0 35px rgba(255,255,255,.035),0 0 0 72px rgba(255,255,255,.025);
    }
    .eyebrow { color:#b9c9ff; font-size:11px; font-weight:700; letter-spacing:.17em; text-transform:uppercase; }
    .hero h1 { font-family:'Playfair Display',serif; font-size:clamp(30px,4vw,43px); margin:9px 0 8px; color:white; }
    .hero p { color:#d5def7; font-size:15px; max-width:650px; line-height:1.65; margin:0; }
    .status-pill {
        display:inline-flex; align-items:center; gap:8px; padding:7px 11px; border:1px solid var(--line);
        border-radius:999px; background:#fff; color:#53627b; font-size:12px; font-weight:600;
    }
    .status-dot { width:8px; height:8px; display:inline-block; border-radius:50%; background:#20ad7b; }
    .section-title { font-family:'Playfair Display',serif; font-size:23px; color:var(--ink); margin:7px 0 3px; }
    .section-subtitle { color:var(--muted); font-size:13px; margin:0 0 18px; }
    .suggestion-label { color:#8792a6; font-size:11px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin:18px 0 8px; }
    div[data-testid="stChatMessage"] {
        border:1px solid var(--line); background:white; border-radius:18px;
        padding:18px 20px; box-shadow:0 4px 16px rgba(27,42,73,.035); margin-bottom:14px;
    }
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h4,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] blockquote,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] em,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] td,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th {
        color:#26344d !important;
    }
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] a {
        color:#3459d1 !important;
    }
    div[data-testid="stChatInput"] { border-color:#dfe4ef; border-radius:17px; box-shadow:0 8px 28px rgba(28,45,82,.08); }
    div[data-testid="stChatInput"] textarea { font-family:'DM Sans',sans-serif; }
    .source-card {
        padding:14px 16px; border:1px solid var(--line); border-radius:14px;
        background:#fbfcff; margin:8px 0 10px;
    }
    .source-meta { color:#63718a !important; font-size:11px; font-weight:700; letter-spacing:.04em; margin-bottom:7px; }
    .source-text { color:#35425a !important; font-size:13px; line-height:1.65; }
    .stButton > button {
        border:1px solid #e1e6f0; border-radius:13px; background:white; color:#33415c;
        min-height:42px; font-weight:600; transition:all .18s ease;
    }
    .stButton > button:hover { border-color:#9caff4; color:#2948b7; background:#f7f9ff; }
    [data-testid="stSidebar"] .stButton > button { border:0; color:#65728a; background:transparent; }
    hr { border-color:var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_rag_resources(api_key: str) -> tuple[OpenAI, QdrantVectorStore]:
    """Create and cache the OpenAI and Qdrant clients for this API key."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
    store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )
    return OpenAI(api_key=api_key), store


@st.cache_data(show_spinner=False)
def get_pdf_page_labels() -> tuple[str, ...]:
    """Return the book's printed page labels for each physical PDF page."""
    reader = PdfReader(str(BASE_DIR / "Basic_german.pdf"))
    labels = reader.page_labels
    if len(labels) != len(reader.pages):
        return tuple(str(index + 1) for index in range(len(reader.pages)))
    return tuple(str(label) for label in labels)


def page_reference(document: Any) -> str:
    """Show the printed book label and physical PDF viewer page."""
    page_index = document.metadata.get("page")
    if not isinstance(page_index, int) or page_index < 0:
        return "Book page unknown · PDF page unknown"

    labels = get_pdf_page_labels()
    if page_index >= len(labels):
        return f"Book page unknown · PDF page {page_index + 1}"

    return f"Book page {labels[page_index]} · PDF page {page_index + 1}"


def build_context(documents: list[Any]) -> str:
    return "\n\n".join(
        f"[Source {index}]\nPDF Content:\n{doc.page_content}\n"
        f"Source file: {doc.metadata.get('source', 'Unknown')}"
        for index, doc in enumerate(documents, start=1)
    )


def answer_question(question: str, api_key: str, result_count: int) -> tuple[str, list[Any]]:
    client, vector_store = get_rag_resources(api_key)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": result_count + 2, "fetch_k": 20, "lambda_mult": 0.5},
    )
    documents = unique_search_results(retriever.invoke(question), limit=result_count)
    if not documents:
        return (
            "I couldn't find relevant passages in the indexed PDF for that question. "
            "Try rephrasing it with a grammar topic or example sentence.",
            [],
        )

    system_prompt = f"""You are an expert, patient German grammar tutor.
Use the retrieved PDF as your primary reference. Explain the concept clearly in
simple English, with practical German examples and translations when useful.
You may supplement the PDF with accurate general German grammar knowledge,
but clearly identify when an explanation goes beyond the PDF. Never invent
PDF citations. When using retrieved material, cite its exact source label, such
as [Source 1]. Do not state book or PDF page numbers; the interface displays
verified page references alongside each source. If the context is not relevant,
say so and distinguish any supplementary explanation from the PDF.

Retrieved PDF context:
{build_context(documents)}"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or "No response was returned.", documents


# Sidebar
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">A</div><div>'
        '<div class="brand-name">Grammatik</div>'
        '<div class="brand-caption">Your German companion</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Learning desk")
    st.caption("Ask a question and explore clear explanations grounded in your German grammar PDF.")
    result_count = st.slider("Reference passages", min_value=2, max_value=6, value=4)
    st.markdown("---")
    st.markdown("**Your study session**")
    st.caption("Answers include source passages and PDF page references whenever available.")
    if st.button("＋  Start a fresh session", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.caption("Powered by OpenAI · Qdrant · your course PDF")

# Hero/header
st.markdown(
    '<div class="hero"><div class="eyebrow">A clearer way to learn</div>'
    '<h1>German grammar, made understandable.</h1>'
    '<p>Ask naturally. Get thoughtful explanations, useful examples, and references '
    'from your learning material—all in one place.</p></div>',
    unsafe_allow_html=True,
)
st.markdown('<span class="status-pill"><span class="status-dot"></span> German grammar tutor ready</span>', unsafe_allow_html=True)
st.markdown('<div class="section-title">What would you like to understand?</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Start with a suggested topic or ask your own question.</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"Explore {len(message['sources'])} PDF references"):
                for index, source in enumerate(message["sources"], start=1):
                    st.markdown(
                        f'<div class="source-card"><div class="source-meta">'
                            f'REFERENCE {index} &nbsp;·&nbsp; {escape(page_reference(source))}'
                        f'</div><div class="source-text">{escape(source.page_content[:1000])}'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

# Suggestions
if not st.session_state.messages:
    st.markdown('<div class="suggestion-label">Popular starting points</div>', unsafe_allow_html=True)
    suggestions = ["When do I use der, die, and das?", "How does German word order work?", "Explain the four German cases."]
    columns = st.columns(3)
    selected_question = None
    for column, suggestion in zip(columns, suggestions):
        if column.button(suggestion, use_container_width=True):
            selected_question = suggestion
else:
    selected_question = None

question = st.chat_input("Ask about German grammar…")
question = question or selected_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        error_message = "Add OPENAI_API_KEY to your .env file, then restart the app."
        with st.chat_message("assistant"):
            st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Finding useful passages and preparing your explanation…"):
                try:
                    answer, sources = answer_question(question, api_key, result_count)
                    st.markdown(answer)
                    if sources:
                        with st.expander(f"Explore {len(sources)} PDF references"):
                            for index, source in enumerate(sources, start=1):
                                st.markdown(
                                    f'<div class="source-card"><div class="source-meta">'
                                    f'REFERENCE {index} &nbsp;·&nbsp; {escape(page_reference(source))}'
                                    f'</div><div class="source-text">{escape(source.page_content[:1000])}'
                                    f'</div></div>',
                                    unsafe_allow_html=True,
                                )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as error:
                    st.error(
                        "Couldn't reach the tutor services. Check that Qdrant is running, "
                        "the collection has been indexed, and your OpenAI key is valid."
                    )
                    st.caption(f"Details: {error}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "Tutor services are unavailable. Check Qdrant, indexing, and the OpenAI key.",
                        }
                    )
