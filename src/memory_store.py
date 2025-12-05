import time
from dataclasses import dataclass, asdict
from typing import List, Literal, Optional, Dict, Any

import faiss
import numpy as np

from .embeddings import embed
from .config import settings

MemoryType = Literal["episodic", "semantic"]


# ------------------------------
# DATA MODEL FOR EACH MEMORY
# ------------------------------
@dataclass
class MemoryItem:
    id: str
    text: str
    memory_type: MemoryType
    created_at: float
    importance: float  # 0–1 scale
    metadata: Dict[str, Any]


# ------------------------------
# VECTOR STORE USING FAISS
# ------------------------------
class FaissMemoryStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner-product similarity
        self.items: List[MemoryItem] = []

    def add(self, item: MemoryItem):
        """Add a memory to the FAISS index."""
        vector = embed(item.text)
        self.index.add(vector)
        self.items.append(item)

    def search(self, query: str, k: int = 5):
        """Retrieve top-k similar memories."""
        q_vector = embed(query)
        scores, indices = self.index.search(q_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = self.items[idx]
            results.append((item, float(score)))
        return results


# ------------------------------
# HUMAN-LIKE MEMORY SYSTEM
# ------------------------------
class HumanLikeMemory:
    """
    Dual-memory architecture:
    • Episodic memory: chronological raw experiences
    • Semantic memory: consolidated summaries

    Weighted recall combines relevance + recency + importance.
    """
    def __init__(self, dim: int = 384):
        self.episodic = FaissMemoryStore(dim)
        self.semantic = FaissMemoryStore(dim)

    # --------------------------
    # REMEMBER
    # --------------------------
    def remember(
        self,
        text: str,
        memory_type: MemoryType = "episodic",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ):
        metadata = metadata or {}
        memory_id = f"{memory_type}-{len(self.episodic.items) + len(self.semantic.items)}"

        item = MemoryItem(
            id=memory_id,
            text=text,
            memory_type=memory_type,
            created_at=time.time(),
            importance=importance,
            metadata=metadata,
        )

        # Store in selected memory type
        store = self.episodic if memory_type == "episodic" else self.semantic
        store.add(item)

        return memory_id

    # --------------------------
    # RECALL
    # --------------------------
    def recall(self, query: str, k: int = 5):
        now = time.time()

        # Search in both memory types
        episodic_hits = self.episodic.search(query, k)
        semantic_hits = self.semantic.search(query, k)

        def weighted_score(item: MemoryItem, similarity: float):
            """
            Compute weighted recall score:
            • relevance (similarity)
            • recency
            • intrinsic importance
            """
            age_seconds = now - item.created_at
            recency = 1.0 / (1.0 + age_seconds / (60 * 60 * 24))  # time decay per day

            score = (
                settings.alpha_relevance * similarity +
                settings.beta_recency * recency +
                settings.gamma_importance * item.importance
            )
            return score

        # Combine and score
        scored_items = []
        for item, sim in episodic_hits + semantic_hits:
            score = weighted_score(item, sim)
            scored_items.append((score, item))

        # Sort by highest score
        scored_items.sort(key=lambda x: x[0], reverse=True)

        # Convert to output format
        return [
            asdict(item) | {"score": score}
            for score, item in scored_items[:k]
        ]

