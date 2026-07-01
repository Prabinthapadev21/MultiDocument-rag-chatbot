import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "app_data.db")

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CHUNK_SIZE_WORDS = 512
CHUNK_OVERLAP_WORDS = 100

EMBEDDING_DIM = 1024
TOP_K_DEFAULT = 4

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS = 1024

PBKDF2_ITERATIONS = 200_000
SALT_BYTES = 16