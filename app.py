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
st.set_page_config(
    page_title="RIT Admission Assistant",
    layout="wide",
    page_icon="🎓"
)

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
    return llm


llm = load_models()


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
# MEMORY (Last 5 Messages)
# =========================
def build_chat_history(messages, limit=5):
    history = messages[-limit:]
    conversation = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation += f"{role}: {msg['content']}\n"
    return conversation


# =========================
# CLEAN CONTEXT
# =========================
def clean_context(text):
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

    return "\n".join(lines)


# =========================
# CHAT SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask your question...")

if query:

    query_lower = query.lower().strip()

    # ================= Casual Chat Handling =================
    greetings = ["hi", "hello", "hii", "hey"]
    confirm_words = ["yes", "yeah", "yup"]
    thanks_words = ["thank", "thanks"]
    ok_words = ["ok", "okay"]

    if query_lower in greetings:
        reply = "Hello 👋 How can I help you today?"
    elif query_lower in confirm_words:
        reply = "Sure 👍 Please tell me your question."
    elif any(word in query_lower for word in thanks_words):
        reply = "You're welcome 😊"
    elif query_lower in ok_words:
        reply = "Alright 👍 Let me know what you'd like to know."
    else:
        reply = None

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        if reply:
            st.markdown(reply)

        else:
            with st.spinner("Analyzing official documents..."):

                nodes = retriever.retrieve(query)

                if nodes:
                    retrieved_text = "\n\n".join([node.text for node in nodes])
                    cleaned_context = clean_context(retrieved_text)
                    chat_history = build_chat_history(st.session_state.messages)

                    prompt = f"""
You are an official RIT Admission Assistant.

STRICT RULES:
- Answer ONLY from the provided context.
- Do NOT add advisory notes.
- Do NOT say "Based on the provided context".
- Do NOT add external knowledge.
- Do NOT guess.
- If information is not clearly available, respond ONLY with:
  "I could not find this information in the official documents provided."
- For cutoff answers, clearly mention Branch, Category, Round, and Marks in bullet format.
- Provide clear, structured answers.

Conversation History:
{chat_history}

Official Context:
{cleaned_context}

User Question:
{query}

Answer:
"""

                    response = llm.complete(prompt)
                    reply = response.text.strip()

                else:
                    reply = "I could not find this information in the official documents provided."

            st.markdown(reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )
