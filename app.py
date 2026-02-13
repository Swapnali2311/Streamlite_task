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
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# CHAT MEMORY (last 5)
# =========================
def get_last_context():
    last_msgs = st.session_state.messages[-5:]
    context = ""
    for msg in last_msgs:
        context += f"{msg['role'].upper()}: {msg['content']}\n"
    return context


# =========================
# GENERATE RESPONSE
# =========================
def generate_response(user_query):

    query_lower = user_query.lower()

    # Natural greetings
    if query_lower in ["hi", "hello", "hii", "hey"]:
        return "Hello 👋 How can I help you today?"

    if query_lower in ["thanks", "thank you", "ok", "okay"]:
        return "You're welcome 😊"

    # Retrieve context
    nodes = retriever.retrieve(user_query)

    if not nodes:
        return "I could not find this information in the official documents."

    context_text = "\n".join([node.text for node in nodes])

    # =========================
    # SMART ELIGIBILITY FIX
    # =========================
    if any(word in query_lower for word in ["eligibility", "eligible", "criteria"]):

        eligibility_lines = []

        for line in context_text.split("\n"):
            if any(keyword in line.lower() for keyword in
                   ["cet", "jee", "merit", "registration", "cap"]):
                eligibility_lines.append(line.strip())

        if eligibility_lines:
            unique = list(dict.fromkeys(eligibility_lines))

            formatted = "### 🎯 Eligibility Criteria for B.Tech\n"
            for item in unique[:6]:
                formatted += f"- {item}\n"

            return formatted
        else:
            return "I could not find this information in the official documents."

    # =========================
    # DEFAULT LLM RESPONSE
    # =========================
    chat_context = get_last_context()

    prompt = f"""
You are an official admission assistant.

Rules:
- Answer ONLY from the provided context.
- Keep response short and structured.
- Do NOT add extra information.
- If not found, say:
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
