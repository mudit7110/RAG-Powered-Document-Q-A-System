import os
import shutil
import streamlit as st

from rag import RAG
from config import DOC_PATH

##############################################
#Streamlit Config
##############################################

st.set_page_config(
    page_title="Documentation RAG",
    page_icon="📚",
    layout="wide"
    )

##############################################
#Cache RAG
##############################################

@st.cache_resource
def load_rag():
    return RAG()

rag = load_rag()

##############################################
#Session State
##############################################

if "messages" not in st.session_state:
    st.session_state.messages = []

##############################################
#Sidebar
##############################################

with st.sidebar:

    st.title("⚙ Settings")

    provider = st.selectbox(
        "LLM Provider",
        [
            "Ollama",
            "OpenAI"
        ]
    )

    st.divider()

    uploaded_files = st.file_uploader(

        "Upload Documents",

        type=[
            "pdf",
            "docx",
            "txt",
            "md"
        ],

        accept_multiple_files=True

    )

    if st.button("📥 Index Documents"):

        if uploaded_files:

            os.makedirs(DOC_PATH, exist_ok=True)

            for file in uploaded_files:

                filepath = os.path.join(
                    DOC_PATH,
                    file.name
                )

                with open(filepath, "wb") as f:

                    f.write(file.read())

            with st.spinner("Building Vector Database..."):

                count = rag.build_database()

            st.success(f"{count} document(s) indexed.")

        else:

            st.warning("Upload documents first.")

    st.divider()

    if st.button("🗑 Reset Database"):

        rag.reset_database()

        st.success("Database cleared.")

    st.divider()

    info = rag.database_info()

    st.subheader("Database")

    st.write(info)

    st.divider()

    st.subheader("Indexed Documents")

    docs = rag.list_documents()

    if docs:

        selected = st.selectbox(

            "Choose",

            docs

        )

        if st.button("Delete Selected"):

            rag.delete_document(selected)

            st.success("Deleted.")

            st.rerun()

    else:

        st.info("No documents indexed.")
##############################################
#Main Page
##############################################

st.title("📚 Documentation Question Answering")

st.caption(
"Powered by ChromaDB + Sentence Transformers + Ollama/OpenAI"
)

##############################################
#Chat History
##############################################

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])
##############################################
#Chat
##############################################

question = st.chat_input(
    "Ask a question..."
)

if question:

    st.session_state.messages.append({

        "role":"user",

        "content":question

    })

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer, sources = rag.ask(

                question,

                provider

            )

        st.markdown(answer)

        with st.expander("Retrieved Sources"):

            for source in sources:

                st.markdown(
f"""
Source: {source['source']}

Chunk: {source['chunk']}

Similarity: {source['similarity']}

{source['text']}

"""
)

    st.session_state.messages.append({

        "role":"assistant",

        "content":answer

    }) 
