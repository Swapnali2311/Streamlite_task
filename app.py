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
Official Admission & Fee Information Portal
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
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_name" not in st.session_state:
    st.session_state.user_name = None


# =========================
# CHAT DISPLAY
# =========================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================
# CHAT INPUT
# =========================
query = st.chat_input("Ask your question...")

if query:

    query_lower = query.lower().strip()

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        # =========================
        # NAME MEMORY
        # =========================
        if "my name is" in query_lower:
            name = query.split("my name is")[-1].strip().capitalize()
            st.session_state.user_name = name
            reply = f"Nice to meet you, {name} 😊 How can I help you today?"

        elif "what is my name" in query_lower or "tell me my name" in query_lower:
            if st.session_state.user_name:
                reply = f"Your name is {st.session_state.user_name} 😊"
            else:
                reply = "You haven’t told me your name yet."

        # =========================
        # NATURAL CHAT RESPONSES
        # =========================
        elif query_lower in ["hi", "hello", "hii", "hey"]:
            reply = "Hello 👋 How can I help you today?"

        elif query_lower in ["thanks", "thank you", "tnks", "thynks"]:
            reply = "You're welcome 😊"

        elif query_lower in ["ok", "okay", "sure", "nice"]:
            reply = "Alright 👍 What would you like to know?"

        elif "can i ask" in query_lower:
            reply = "Of course 😊 Please go ahead."

        # =========================
        # DOCUMENT-BASED RESPONSE
        # =========================
        else:
            with st.spinner("Searching official documents..."):

                nodes = retriever.retrieve(query)

                if nodes:
                    context = "\n\n".join([node.text for node in nodes])

                    prompt = f"""
You are an official RIT Admission Assistant.

STRICT RULES:
- Answer ONLY from the provided context.
- Keep answers SHORT and clear.
- Do NOT repeat unnecessary information.
- Do NOT give advisory notes.
- Do NOT add external knowledge.
- If not found, say:
"I could not find this information in the official documents."

Context:
{context}

Question:
{query}

Answer:
"""

                    response = llm.complete(prompt)
                    reply = response.text.strip()

                else:
                    reply = "I could not find this information in the official documents."

        st.markdown(reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )
