import os 
from dotenv import load_dotenv

load_dotenv()
###########################################################

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

###########################################################

CHROMA_PATH = "data/chroma"

DOC_PATH = "data/docs"

###########################################################

TOP_K = 4
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75

############################################################

OLLAMA_MODEL = "qwen2.5:3b"
OPENAI_MODEL = "gpt-5.5"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
