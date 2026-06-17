import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Embedding Configurations
EMBEDDING_MODEL_NAME = "dunzhang/stella_en_400M_v5"
EMBEDDING_TRUST_REMOTE_CODE = True
EMBEDDING_PROMPT_NAME = "s2p_query"

# Text Splitting Configurations
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval Configurations
RETRIEVAL_K = 4
SIMILARITY_SCORE_THRESHOLD = 0.35

# LLM Configurations
GROQ_MODEL_NAME = "llama-3.1-8b-instant"   # Replaces decommissioned llama3-8b-8192
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 512

# File Paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
SYSTEM_PROMPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "prompts", "system_prompt.txt"))

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)
