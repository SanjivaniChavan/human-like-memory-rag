from fastapi import FastAPI
from pydantic import BaseModel

from .memory_store import HumanLikeMemory
from .summarizer import simple_summarize

app = FastAPI(title="Human-Like Memory API")

memory = HumanLikeMemory()

class RememberRequest(BaseModel):
    text: str
    memory_type: str = "episodic"
    importance: float = 0.5

class RecallRequest(BaseModel):
    query: str
    k: int = 5


@app.post("/remember")
def remember(req: RememberRequest):
    mem_id = memory.remember(
        text=req.text,
        memory_type=req.memory_type,
        importance=req.importance
    )
    return {"id": mem_id}


@app.post("/recall")
def recall(req: RecallRequest):
    memories = memory.recall(req.query, k=req.k)
    summary = simple_summarize(memories)
    return {
        "memories": memories,
        "summary": summary
    }
