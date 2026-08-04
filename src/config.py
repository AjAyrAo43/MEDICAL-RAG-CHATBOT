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
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"]     = GROQ_API_KEY

# ── Zero-RAM Cloud Embeddings Client ─────────────────
from langchain_core.embeddings import Embeddings
import requests

class CloudInferenceEmbeddings(Embeddings):
    """
    Zero-RAM Cloud Embeddings client for BAAI/bge-large-en.
    Queries Hugging Face Inference API over HTTPS to prevent OOM status 137 on Render Free Tier.
    """
    def __init__(self, model_name="BAAI/bge-large-en", api_key=None):
        self.url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.api_key = api_key

    def _request(self, payload):
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        res = requests.post(self.url, headers=headers, json=payload, timeout=30)
        if res.status_code != 200:
            raise RuntimeError(f"HuggingFace API Error ({res.status_code}): {res.text}")
        return res.json()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        data = self._request({"inputs": texts, "options": {"wait_for_model": True}})
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            if len(data[0]) > 0 and isinstance(data[0][0], list):
                return [[sum(col) / len(col) for col in zip(*doc)] for doc in data]
            return data
        return data

    def embed_query(self, text: str) -> list[float]:
        data = self._request({"inputs": text, "options": {"wait_for_model": True}})
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], list):
                return [sum(col) / len(col) for col in zip(*data)]
            return data
        return data

# ── Embeddings Selection ───────────────────────
hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
is_cloud = os.getenv("RENDER") is not None or os.getenv("PORT") is not None

if is_cloud:
    print("Using zero-RAM CloudInferenceEmbeddings for Render deployment...")
    embeddings = CloudInferenceEmbeddings("BAAI/bge-large-en", api_key=hf_token)
else:
    try:
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        print(f"Local BGE embeddings failed ({e}), using CloudInferenceEmbeddings...")
        embeddings = CloudInferenceEmbeddings("BAAI/bge-large-en", api_key=hf_token)

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