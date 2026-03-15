# ─────────────────────────────────────────────
# config.py — Global setup: env, device, LLM, embeddings, vectorstore
# ─────────────────────────────────────────────
import os
import torch
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# ── Device ────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ── API Keys ──────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"]     = GROQ_API_KEY

# ── Embeddings ────────────────────────────────
# ✅ Must match the model used when data was originally stored
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-en",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)

# ── Pinecone Vectorstore ──────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)

vectorstore = PineconeVectorStore(
    index_name="medical-index",
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY
)

# ── LLM ──────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)