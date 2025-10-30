"""Embeddings utilities (optional)."""


def embed_text_placeholder(text: str) -> list:
    # return fixed-length dummy embedding
    return [len(text) % 7] * 128
