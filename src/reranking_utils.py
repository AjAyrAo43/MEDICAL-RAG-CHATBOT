# ─────────────────────────────────────────────
# reranking_utils.py — Cross-Encoder re-ranking after RRF fusion
# ─────────────────────────────────────────────
from sentence_transformers import CrossEncoder

# Load once at module level (avoids repeated model loading)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_with_cross_encoder(query: str, docs: list, top_n: int = 5) -> list:
    """
    Re-scores RRF-fused documents using a cross-encoder for fine-grained ranking.

    Cross-encoders jointly encode (query, document) pairs and produce a single
    relevance score — far more accurate than cosine similarity on embeddings alone.

    Args:
        query  : The user's (possibly rephrased) question
        docs   : Candidate documents from RRF fusion
        top_n  : Number of top documents to return

    Returns:
        List of top_n Documents sorted by cross-encoder relevance score
    """
    if not docs:
        return []

    pairs  = [[query, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs)

    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_n]]