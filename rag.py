import os
from tkinter.font import names
import uuid
import hashlib
import re
from pathlib import Path

import chromadb

from chromadb.config import Settings

from sentence_transformers import SentenceTransformer

from pypdf import PdfReader

from docx import Document, text

from config import *
import llm

class RAG:

##########################################################
# Initialization
##########################################################

    def __init__(self):

        self.embedding_model = SentenceTransformer(
            EMBED_MODEL
        )

        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        self.collection = self.client.get_or_create_collection(
            name="documents"
        )

##########################################################
# Read Files
##########################################################

def read_pdf(self, filepath):

    reader = PdfReader(filepath)

    text = ""

    for page in reader.pages:

        content = page.extract_text()

        if content:
            text += content + "\n"

    return text


def read_docx(self, filepath):

    doc = Document(filepath)

    paragraphs = []

    for p in doc.paragraphs:
        paragraphs.append(p.text)

    return "\n".join(paragraphs)


def read_txt(self, filepath):

    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def read_md(self, filepath):

    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

##########################################################
# Read according to extension
##########################################################

def load_document(self, filepath):

    extension = Path(filepath).suffix.lower()

    if extension == ".pdf":
        return self.read_pdf(filepath)

    elif extension == ".docx":
        return self.read_docx(filepath)

    elif extension == ".txt":
        return self.read_txt(filepath)

    elif extension == ".md":
        return self.read_md(filepath)

    else:
        raise Exception(f"Unsupported file : {extension}")

##########################################################
# Chunking
##########################################################

def chunk_text(self, text):

    # Normalize whitespace
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Split into paragraphs first
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # If paragraph fits, append it
        if len(current_chunk) + len(paragraph) <= CHUNK_SIZE:

            current_chunk += paragraph + "\n\n"

        else:

            if current_chunk:
                chunks.append(current_chunk.strip())

            # Long paragraph? Split into sentences
            if len(paragraph) > CHUNK_SIZE:

                sentences = re.split(
                    r'(?<=[.!?])\s+',
                    paragraph
                )

                current_chunk = ""

                for sentence in sentences:

                    if len(current_chunk) + len(sentence) <= CHUNK_SIZE:

                        current_chunk += sentence + " "

                    else:

                        chunks.append(current_chunk.strip())

                        overlap = current_chunk[-CHUNK_OVERLAP:]

                        current_chunk = overlap + sentence + " "

            else:

                current_chunk = paragraph + "\n\n"

    if current_chunk:

        chunks.append(current_chunk.strip())

    return chunks

##########################################################
# Hash
##########################################################

def file_hash(self, filepath):

    md5 = hashlib.md5()

    with open(filepath, "rb") as f:

        while True:

            data = f.read(4096)

            if not data:
                break

            md5.update(data)

    return md5.hexdigest()

##########################################################
# Duplicate Check
##########################################################

def already_exists(self, filehash):

    result = self.collection.get(
        where={
            "file_hash": filehash
        }
    )

    return len(result["ids"]) > 0

##########################################################
# Embed
##########################################################

def embed(self, chunks):

    return self.embedding_model.encode(
        chunks,
        show_progress_bar=True
    ).tolist()

##########################################################
# Index One File
##########################################################

def index_document(self, filepath):

    filehash = self.file_hash(filepath)

    if self.already_exists(filehash):

        return False

    text = self.load_document(filepath)

    chunks = self.chunk_text(text)

    embeddings = self.embed(chunks)

    ids = []

    metadatas = []

    documents = []

    for i, chunk in enumerate(chunks):

        ids.append(str(uuid.uuid4()))

        documents.append(chunk)

        metadatas.append({

            "source": os.path.basename(filepath),

            "chunk": i,

            "file_hash": filehash

        })

    self.collection.add(

        ids=ids,

        embeddings=embeddings,

        documents=documents,

        metadatas=metadatas

    )

    return True

##########################################################
# Index Folder
##########################################################

def build_database(self):

    count = 0

    for file in os.listdir(DOC_PATH):

        path = os.path.join(
            DOC_PATH,
            file
        )

        if os.path.isfile(path):

            try:

                if self.index_document(path):
                    count += 1

            except Exception as e:

                print(e)

    return count

##########################################################
# Stats
##########################################################

def total_chunks(self):

    return self.collection.count()

##########################################################
# Search
##########################################################

def search(self, question):
    embedding = self.embedding_model.encode(
    question
    ).tolist()

    results = self.collection.query(

        query_embeddings=[embedding],

        n_results=TOP_K

    )

    retrieved = []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, distance in zip(
        docs,
        metas,
        distances
    ):

        similarity = 1 / (1 + distance)

        retrieved.append({

            "text": doc,

            "source": meta["source"],

            "chunk": meta["chunk"],

            "distance": distance,

            "similarity": round(
                similarity,
                3
            )

        })

    return retrieved

##########################################################
# List Documents
##########################################################

def list_documents(self):
    results = self.collection.get()

    names = set()

    for meta in results["metadatas"]:

        names.add(meta["source"])

    return sorted(list(names))

##########################################################
# Delete Documents
##########################################################

def delete_document(self, filename):
    results = self.collection.get()

    ids = []

    for i, meta in enumerate(results["metadatas"]):

        if meta["source"] == filename:

            ids.append(results["ids"][i])

    if ids:

        self.collection.delete(
            ids=ids
        )

        return True

    return False

##########################################################
# Delete Documents
##########################################################

def database_info(self):
    return{
        "Collection":"Documents",
        "Chunks": self.collection.count(),
        "Documents": len(self.list_documents()),
        "Embedding Model": EMBED_MODEL,
    }

##########################################################
# Reset Database
##########################################################

def reset_database(self):
    try:
        self.client.delete_collection("documents")
    except Exception:
        pass

    self.collection = self.client.get_or_create_collection(
        name="documents")
