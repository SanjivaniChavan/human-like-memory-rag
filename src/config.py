from pydantic import BaseSettings

class Settings(BaseSettings):
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    episodic_faiss_index_path: str = "data/episodic.index"
    semantic_faiss_index_path: str = "data/semantic.index"
    alpha_relevance: float = 0.6
    beta_recency: float = 0.25
    gamma_importance: float = 0.15

settings = Settings()
