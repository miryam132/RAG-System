"""הגדרות מרכזיות של המערכת: מודלים, נתיבים ופרמטרים.

כל המספרים שכדאי לשחק איתם (chunk_size, top_k וכו') נמצאים כאן,
כדי שבשלב הבדיקות נוכל לשנות אותם במקום אחד.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()  # קורא את GOOGLE_API_KEY מקובץ .env

# נתיבים
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "board_games"

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Retrieval
SIMILARITY_TOP_K = 4

# מודלים
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
LLM_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def setup_embeddings() -> None:
    """מגדיר את מודל ה-embeddings (מקומי, לא דורש מפתח)."""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        # מודלי e5 דורשים קידומת שמבדילה בין שאלה לבין מסמך
        query_instruction="query: ",
        text_instruction="passage: ",
    )


def setup_llm() -> None:
    """מגדיר את מודל השפה (Gemini, דורש GOOGLE_API_KEY)."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("חסר GOOGLE_API_KEY בקובץ .env")
    Settings.llm = GoogleGenAI(model=LLM_MODEL_NAME, temperature=0.1)
