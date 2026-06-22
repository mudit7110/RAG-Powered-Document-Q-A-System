import os
import tempfile
from typing import List

import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Optional: OpenAI
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    ChatOpenAI = None
    OpenAIEmbeddings = None

st.set_page_config(page_title="RAG in a box", page_icon="📚")

st.title("📚 RAG system: upload a doc, ask anything")

# Sidebar for model choices
with st.sidebar:
    st.header("Settings")
    
    llm_choice = st.radio(
        "LLM to use",
        ["Open-source (Hugging Face)", "OpenAI"],
        help="Open-source runs locally (free). OpenAI needs an API key."
    )
    
    if llm_choice == "OpenAI":
        api_key = st.text_input("OpenAI API key", type="password")
        model_name = st.text_input("Model", value="gpt-4o-mini")
    else:
        # Open source defaults
        hf_model = st.text_input(
            "Hugging Face chat model",
            value="mistralai/Mistral-7B-Instruct-v0.3",
            help="Any instruct/chat model on HF (e.g. meta-llama/Llama-3-8B-Instruct, Qwen/Qwen3-8B)"
        )
        max_new_tokens = st.slider("Max new tokens", 256, 2048, 1024, 128)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    
    embed_choice = st.radio(
        "Embedding model",
        ["sentence-transformers (local, free)", "OpenAI embeddings"],
    )
    
    if embed_choice == "OpenAI embeddings":
        if llm_choice != "OpenAI":
            embed_key = st.text_input("OpenAI API key (for embeddings)", type="password")
        else:
            embed_key = api_key
        embed_model = st.text_input("Embedding model", value="text-embedding-3-small")
    else:
        embed_model = st.text_input(
            "SentenceTransformer model",
            value="BAAI/bge-small-en-v1.5",
            help="Great general-purpose small embedding model"
        )
    
    chunk_size = st.slider("Chunk size", 256, 2048, 800, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 100, 50)
    
    st.caption("Everything runs locally for embeddings + open-source LLM. Your docs stay on your machine.")


# File uploader
uploaded = st.file_uploader(
    "Upload a document",
    type=["pdf", "txt", "docx", "csv", "md"],
    accept_multiple_files=False
)

# Helper: load docs
def load_document(file):
    # Save to temp file
    ext = file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(file.read())
        path = tmp.name
    
    try:
        if ext == "pdf":
            loader = PyPDFLoader(path)
        elif ext == "txt" or ext == "md":
            loader = TextLoader(path, encoding="utf-8")
        elif ext == "docx":
            loader = Docx2txtLoader(path)
        elif ext == "csv":
            loader = CSVLoader(path)
        else:
            # Fallback
            loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        # Add source metadata
        for d in docs:
            d.metadata["source"] = file.name
        return docs
    finally:
        # Clean up
        try:
            os.unlink(path)
        except:
            pass


# Helper: get embeddings
@st.cache_resource
def get_embeddings(choice: str, model_name: str, api_key: str | None = None):
    if choice == "OpenAI embeddings":
        if OpenAIEmbeddings is None:
            raise RuntimeError("langchain-openai not installed")
        if not api_key:
            raise RuntimeError("Missing OpenAI API key for embeddings")
        return OpenAIEmbeddings(model=model_name, api_key=api_key)
    else:
        # SentenceTransformers
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},  # change to "cuda" if you have GPU
        )


# Helper: get LLM
def get_llm(choice: str, **kwargs):
    if choice == "OpenAI":
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai not installed")
        api_key = kwargs.get("api_key")
        if not api_key:
            raise RuntimeError("Missing OpenAI API key")
        return ChatOpenAI(model=kwargs.get("model", "gpt-4o-mini"), api_key=api_key, temperature=0.2)
    else:
        # Open source via Hugging Face pipeline
        from langchain_community.llms import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch
        
        model_id = kwargs.get("hf_model", "mistralai/Mistral-7B-Instruct-v0.3")
        
        # Load tokenizer + model
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Try efficient loading
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=kwargs.get("max_new_tokens", 1024),
            temperature=kwargs.get("temperature", 0.2),
            do_sample=kwargs.get("temperature", 0.2) > 0,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        
        return HuggingFacePipeline(pipeline=pipe)


# Build chain
def build_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Prompt: be grounded, cite sources if possible
    template = """
You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer isn't in the context, say "I don't have that in the document."

Context:
{context}

Question:
{question}

Answer (grounded):
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs: List[Document]) -> str:
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}, page {d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        )
    
    # Build the chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


# State
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None


# Process document
if uploaded:
    with st.spinner("Processing document..."):
        # 1. Load
        raw_docs = load_document(uploaded)
        if not raw_docs:
            st.error("Couldn't load document.")
            st.stop()
        
        # 2. Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(raw_docs)
        
        st.success(f"Loaded {len(raw_docs)} pages/rows → {len(chunks)} chunks")
        
        # 3. Embeddings
        try:
            if embed_choice == "OpenAI embeddings":
                key = embed_key if 'embed_key' in locals() or 'embed_key' in globals() else api_key
                embeddings = get_embeddings(embed_choice, embed_model, key)
            else:
                embeddings = get_embeddings(embed_choice, embed_model)
        except Exception as e:
            st.error(f"Failed to load embeddings: {e}")
            st.stop()
        
        # 4. Vector store
        try:
            # Build FAISS
            vectorstore = FAISS.from_documents(chunks, embeddings)
            st.session_state.vectorstore = vectorstore
        except Exception as e:
            st.error(f"Failed to build vector store: {e}")
            st.stop()
        
        # 5. LLM + chain
        try:
            if llm_choice == "OpenAI":
                llm = get_llm("OpenAI", api_key=api_key, model=model_name)
            else:
                llm = get_llm(
                    "Open-source (Hugging Face)",
                    hf_model=hf_model,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            st.session_state.chain = build_rag_chain(vectorstore, llm)
        except Exception as e:
            st.error(f"Failed to load LLM: {e}")
            st.info("Tip: for open-source, first run may download the model (a few GB). Be patient.")
            st.stop()
        
        st.success("Ready to answer questions! Ask below.")

# Chat
if st.session_state.chain is None:
    st.info("Upload a document to get started.")
else:
    q = st.text_input("Ask a question about your document", placeholder="e.g. What are the key takeaways?")
    if q:
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.chain.invoke(q)
                st.markdown("### Answer")
                st.write(answer)
                
                # Show sources
                with st.expander("Sources used"):
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
                    docs = retriever.invoke(q)
                    for i, d in enumerate(docs, 1):
                        src = d.metadata.get("source", "unknown")
                        page = d.metadata.get("page", "?")
                        st.markdown(f"**{i}. {src} (page {page})**")
                        st.code(d.page_content[:800] + ("..." if len(d.page_content) > 800 else ""))
                
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# Footer
st.markdown("---")
st.caption("Built with LangChain + FAISS + sentence-transformers + Streamlit. Uses your local machine by default.")
