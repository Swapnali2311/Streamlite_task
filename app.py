import streamlit as st
import os

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.readers.file import PyMuPDFReader


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="RIT Admission Assistant", layout="wide")

st.markdown("""
<h1 style='text-align:center;'>🎓 RIT Admission Assistant</h1>
<p style='text-align:center;color:gray;'>
Official Admission & Fee Information Portal (RAG Powered)
</p>
""", unsafe_allow_html=True)

st.divider()


# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.embed_model = embed_model

    llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    Settings.llm = llm

load_models()


# =========================
# LOAD INDEX
# =========================
@st.cache_resource
def load_index():
    if os.path.exists("storage"):
        storage_context = StorageContext.from_defaults(persist_dir="storage")
        index = load_index_from_storage(storage_context)
    else:
        reader = PyMuPDFReader()
        documents = []

        for file in os.listdir("admission_docs"):
            if file.endswith(".pdf"):
                documents.extend(
                    reader.load_data(os.path.join("admission_docs", file))
                )

        splitter = SentenceSplitter(chunk_size=800, chunk_overlap=80)

        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[splitter]
        )

        index.storage_context.persist(persist_dir="storage")

    return index


index = load_index()
retriever = index.as_retriever(similarity_top_k=8)


# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "username" not in st.session_state:
    st.session_state.username = None


# =========================
# CHAT MEMORY
# =========================
def get_last_context():
    last_msgs = st.session_state.messages[-5:]
    context = ""
    for msg in last_msgs:
        context += f"{msg['role'].upper()}: {msg['content']}\n"
    return context


# =========================
# RESPONSE GENERATION
# =========================
def generate_response(user_query):

    query_lower = user_query.lower().strip()

    # -------------------------
    # Greeting + Name Handling
    # -------------------------
    if any(greet in query_lower for greet in ["hi", "hello", "hii", "hey"]):

        if "i am" in query_lower:
            name = query_lower.split("i am")[-1].strip().title()
            st.session_state.username = name
            return f"Hello {name} 👋 How can I help you regarding admissions?"

        if st.session_state.username:
            return f"Welcome back {st.session_state.username} 👋 How can I help you?"

        return "Hello 👋 How can I help you regarding admissions?"

    # -------------------------
    # Thanks Handling
    # -------------------------
    if query_lower in ["thanks", "thank you", "ok", "okay", "great", "awesome"]:
        return "You're welcome 😊"

    # -------------------------
    # Special Case: 10th Pass
    # -------------------------
    if "10th" in query_lower:
        return (
            "### Eligibility Status\n\n"
            "❌ You are not eligible for B.Tech admission.\n\n"
            "Candidate must complete 12th (HSC) with Physics, Chemistry and Mathematics before applying."
        )

    # -------------------------
    # Retrieve Context
    # -------------------------
    nodes = retriever.retrieve(user_query)

    if not nodes:
        return "I could not find this information in the official documents."

    context_text = "\n".join([node.text for node in nodes])

    # -------------------------
    # Eligibility Criteria
    # -------------------------
    if any(word in query_lower for word in ["eligibility", "criteria"]):
        return (
            "### Eligibility Criteria for B.Tech\n\n"
            "- Candidate must have completed 12th (HSC) with Physics, Chemistry and Mathematics.\n"
            "- Admission is based on CET / JEE merit.\n"
            "- Registration under CAP or Institute Level quota is required."
        )

    # -------------------------
    # Default LLM (Strict RAG)
    # -------------------------
    chat_context = get_last_context()

    prompt = f"""
You are an official admission assistant.

Strict Rules:
- Answer ONLY from the provided context.
- Keep answer short and structured.
- Do NOT add extra information.
- If answer not found in context, say:
"I could not find this information in the official documents."

Conversation:
{chat_context}

Context:
{context_text}

Question:
{user_query}

Answer:
"""

    response = Settings.llm.complete(prompt)
    return response.text.strip()


# =========================
# CHAT UI
# =========================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask your question...")

if user_input:

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Fetching official information..."):
            reply = generate_response(user_input)
            st.markdown(reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )
