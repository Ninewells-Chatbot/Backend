from fastapi import FastAPI
from pydantic import BaseModel
from rag import ask_rag
from memory import get_history, add_message
import uuid

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


@app.post("/chat")
def chat(req: ChatRequest):

    # Create session if new
    if not req.session_id:
        req.session_id = str(uuid.uuid4())

    history = get_history(req.session_id)

    # Add user message
    add_message(req.session_id, "user", req.message)

    # Get AI response
    answer = ask_rag(req.message, history)

    # Save assistant response
    add_message(req.session_id, "assistant", answer)

    return {
        "session_id": req.session_id,
        "response": answer
    }
