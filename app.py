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
    <p style='text-align:center; color:gray;'>
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

query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="compact"
)


# =========================
# CHAT MEMORY (Last 5 Messages)
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_last_context():
    last_msgs = st.session_state.messages[-5:]
    context = ""
    for msg in last_msgs:
        role = msg["role"]
        content = msg["content"]
        context += f"{role.upper()}: {content}\n"
    return context


# =========================
# RESPONSE FORMATTER
# =========================
def generate_response(user_query):

    query_lower = user_query.lower()

    # Natural greetings
    greetings = ["hi", "hello", "hii", "hey"]
    if query_lower in greetings:
        return "Hello 👋 How can I help you with admission or fee information today?"

    casual = ["thanks", "thank you", "ok", "okay"]
    if query_lower in casual:
        return "You're welcome 😊 Let me know if you need any more information."

    # Build context-aware prompt
    chat_context = get_last_context()

    strict_prompt = f"""
You are an official admission assistant.

Rules:
1. Answer ONLY using the provided context.
2. Do NOT assume or add extra information.
3. If answer is not found, say:
   "I could not find this information in the official documents."
4. Keep response clear and structured.
5. Avoid mentioning page numbers, signatures, stamps, website, or contact details.

Conversation context:
{chat_context}

User Question:
{user_query}
"""

    try:
        response = query_engine.query(strict_prompt)
        answer = str(response)

        if not answer.strip():
            return "I could not find this information in the official documents."

        return answer

    except:
        return "I could not find this information in the official documents."


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
