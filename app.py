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


# =========================
# CHAT MEMORY (last 5 msgs)
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

    query_lower = user_query.lower()

    # -------------------------
    # Natural Greetings
    # -------------------------
    if query_lower in ["hi", "hello", "hii", "hey"]:
        return "Hello 👋 How can I help you regarding admissions?"

    if query_lower in ["thanks", "thank you", "ok", "okay"]:
        return "You're welcome 😊"


    # -------------------------
    # Retrieve Context
    # -------------------------
    nodes = retriever.retrieve(user_query)

    if not nodes:
        return "I could not find this information in the official documents."

    context_text = "\n".join([node.text for node in nodes])


    # =========================================================
    # 🔥 SMART ELIGIBILITY HANDLING (Logic Based Answering)
    # =========================================================
    if any(word in query_lower for word in ["eligibility", "eligible", "apply", "can i", "admission"]):

        eligibility_lines = []

        for line in context_text.split("\n"):
            line_clean = line.strip()

            if (
                "seat no" in line_clean.lower()
                or "☐" in line_clean
                or "____" in line_clean
                or len(line_clean) < 15
            ):
                continue

            if any(word in line_clean.lower() for word in 
                   ["12th", "hsc", "pcm", "physics", "chemistry", "mathematics"]):
                eligibility_lines.append(line_clean)

        if eligibility_lines:

            unique_lines = list(dict.fromkeys(eligibility_lines))

            # 🔴 Special case: 10th pass query
            if "10th" in query_lower:
                return (
                    "### Eligibility Status\n\n"
                    "❌ You are not eligible for B.Tech admission.\n\n"
                    "As per official criteria, candidate must have completed "
                    "12th (HSC) with Physics, Chemistry and Mathematics.\n\n"
                    "Please complete 12th standard before applying."
                )

            formatted = "### Eligibility Criteria for B.Tech\n\n"
            for line in unique_lines[:3]:
                formatted += f"- {line}\n"

            return formatted

        return "I could not find this information in the official documents."


    # =========================================================
    # DEFAULT LLM RESPONSE (STRICT RAG MODE)
    # =========================================================
    chat_context = get_last_context()

    prompt = f"""
You are an official admission assistant.

Strict Rules:
- Answer ONLY from the provided context.
- Keep answer short and structured.
- If qualification does not meet criteria, clearly say NOT ELIGIBLE.
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
