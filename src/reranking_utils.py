# ─────────────────────────────────────────────
# reranking_utils.py — Cross-Encoder re-ranking after RRF fusion
# ─────────────────────────────────────────────

_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank_with_cross_encoder(query: str, docs: list, top_n: int = 5) -> list:
    """
    Re-scores RRF-fused documents using a cross-encoder for fine-grained ranking.
    """
    if not docs:
        return []

    try:
        encoder = get_cross_encoder()
        pairs  = [[query, doc.page_content] for doc in docs]
        scores = encoder.predict(pairs)
        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_n]]
    except Exception as e:
        print(f"Cross-encoder reranking skipped due to error: {e}")
        return docs[:top_n]