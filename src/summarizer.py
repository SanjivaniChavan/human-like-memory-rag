from typing import List

def simple_summarize(memories: List[dict]) -> str:
    """
    Very simple placeholder summarizer.
    Later you can replace with a real LLM model or a fine-tuned summarizer.
    """
    texts = [m["text"] for m in memories]
    
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]
    
    # Combine all memory texts and truncate
    combined = " ".join(texts)
    return combined[:1000]  # placeholder
