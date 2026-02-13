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
st.set_page_config(page_title="RIT Admission Assistant", layout="centered")

st.markdown("""
    <h1 style='text-align: center;'>🎓 RIT Admission Assistant</h1>
    <p style='text-align: center; color: gray;'>
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
# LOAD / BUILD INDEX
# =========================
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

        splitter = SentenceSplitter(chunk_size=700, chunk_overlap=40)

        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[splitter]
        )

        index.storage_context.persist(persist_dir="storage")

    return index


index = load_index()
retriever = index.as_retriever(similarity_top_k=5)


# =========================
# FORMAT ANSWER FUNCTION
# =========================
def format_answer(query, text):

    query_lower = query.lower()

    # Handle casual messages
    casual_words = ["thank", "thanks", "ok", "okay", "hi", "hello"]
    if any(word in query_lower for word in casual_words):
        return "You're welcome 😊 Let me know if you need any admission-related information."

    remove_words = [
        "paragraph format",
        "page",
        "authorized signatory",
        "signature",
        "stamp",
        "institute contact",
        "phone",
        "email",
        "website",
        "declaration",
        "fees (₹)",
        "total fees",
        "sr. no.",
        "description"
    ]

    lines = []
    for line in text.split("\n"):
        clean = line.strip()
        if clean and not any(word in clean.lower() for word in remove_words):
            lines.append(clean)

    # ========= REQUIRED DOCUMENTS =========
    if "document" in query_lower:
        docs = [
            l for l in lines
            if any(keyword in l.lower() for keyword in
                   ["certificate", "mark", "score", "leaving"])
        ]

        unique_docs = list(dict.fromkeys(docs))

        formatted = "### 📄 Required Documents\n"
        for doc in unique_docs:
            formatted += f"- {doc}\n"

        return formatted


    # ========= CUTOFF =========
    if "cutoff" in query_lower or "cut off" in query_lower:

        cutoff_lines = [
            l for l in lines
            if "computer engineering" in l.lower()
        ]

        unique_cutoff = list(dict.fromkeys(cutoff_lines))

        formatted = "### 📊 Cutoff Information\n"
        for c in unique_cutoff:
            formatted += f"- {c}\n"

        return formatted


    # ========= FEE STRUCTURE =========
    if "fee" in query_lower:

        fee_lines = [
            l for l in lines
            if "₹" in l
        ]

        unique_fees = list(dict.fromkeys(fee_lines))

        formatted = "### 💰 Fee Details\n"
        for fee in unique_fees:
            formatted += f"- {fee}\n"

        return formatted


    # ========= ADMISSION CRITERIA =========
    if "criteria" in query_lower and "cut" not in query_lower:

        criteria_lines = [
            l for l in lines
            if any(keyword in l.lower() for keyword in
                   ["cet", "jee", "merit", "quota"])
        ]

        unique_criteria = list(dict.fromkeys(criteria_lines))

        formatted = "### 🎯 Admission Criteria\n"
        for c in unique_criteria:
            formatted += f"- {c}\n"

        return formatted

    return "Information available in official documents."


# =========================
# CHAT UI
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask your question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Fetching official information..."):
            nodes = retriever.retrieve(query)

            if nodes:
                formatted_answer = format_answer(query, nodes[0].text)
            else:
                formatted_answer = "Information not available in official documents."

            st.markdown(formatted_answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": formatted_answer}
    )
