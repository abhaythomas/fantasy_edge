"""
api/api.py — FastAPI Endpoint for FantasyEdge

Serves the agentic FPL analyst as a REST API.
Supports conversational queries with session management.

Run with: uvicorn api.api:app --reload
Docs at: http://localhost:8000/docs
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from agent.graph import chat

app = FastAPI(
    title="FantasyEdge API",
    description="Agentic FPL analyst powered by LangGraph ReAct. "
                "Ask any FPL question and the agent autonomously decides "
                "which tools to use to answer it.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple session storage (in production, use Redis or a database)
_sessions = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    response: str
    tool_calls: list
    processing_time: float
    session_id: str


@app.get("/")
def root():
    return {
        "name": "FantasyEdge API",
        "version": "2.0.0",
        "description": "Agentic FPL analyst — ask any FPL question",
        "endpoints": {
            "POST /chat": "Send a message to the agent",
            "GET /docs": "Interactive API documentation",
            "GET /health": "Health check",
        },
    }


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Send a message to the FantasyEdge agent.
    The agent autonomously decides which tools to use and responds.
    Use session_id for conversation continuity across requests.
    """
    try:
        start = time.time()

        # Get conversation history for this session
        history = _sessions.get(request.session_id)

        result = chat(request.message, history)

        # Save history for continuity
        _sessions[request.session_id] = result["messages"]

        elapsed = round(time.time() - start, 1)

        return ChatResponse(
            response=result["response"],
            tool_calls=result["tool_calls"],
            processing_time=elapsed,
            session_id=request.session_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clear conversation history for a session."""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"message": f"Session '{session_id}' cleared"}
    return {"message": f"Session '{session_id}' not found"}


@app.get("/health")
def health():
    return {"status": "healthy", "agent": "FantasyEdge v2.0"}
