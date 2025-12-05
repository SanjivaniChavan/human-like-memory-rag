from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    model = get_model()
    embs = model.encode(texts, normalize_embeddings=True)
    return np.array(embs, dtype="float32")
