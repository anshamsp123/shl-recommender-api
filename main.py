from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal
import asyncio
import logging

from catalog import CatalogIndex, get_catalog
from agent import is_out_of_scope, synthesize_query, call_agent

app = FastAPI(title="SHL Conversational Assessment Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = []
    end_of_conversation: bool = False

@app.on_event("startup")
async def startup_event():
    try:
        # Pre-load the catalog and index
        get_catalog()
    except Exception as e:
        logging.error(f"Startup error: {e}")
        # The prompt says: "Catalog JSON missing or empty at startup -> raise RuntimeError with clear message, do not start server"
        raise RuntimeError(f"Failed to initialize catalog: {e}")

@app.get("/health")
async def health():
    return {"status": "ok"}

def validate_and_build_response(response: dict, index: CatalogIndex) -> ChatResponse:
    reply = response.get("reply", "Sorry, I encountered an error. Please try again.")
    raw_recs = response.get("recommendations", [])
    end_of_conv = response.get("end_of_conversation", False)
    
    validated_recs = []
    for rec in raw_recs:
        name = rec.get("name")
        # Validate name exists exactly
        catalog_entry = index.get_by_name(name)
        if not catalog_entry:
            logging.warning(f"Hallucinated assessment dropped: {name}")
            continue
            
        # Replace URL and test_type with catalog values
        url = catalog_entry["url"]
        test_type = catalog_entry.get("test_type", "")
        
        validated_recs.append(Recommendation(
            name=name,
            url=url,
            test_type=test_type
        ))
        
        if len(validated_recs) >= 10:
            break
            
    return ChatResponse(
        reply=reply,
        recommendations=validated_recs,
        end_of_conversation=end_of_conv
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, index: CatalogIndex = Depends(get_catalog)):
    messages = request.messages
    
    if not messages:
        raise HTTPException(status_code=422, detail="messages must not be empty")
        
    if messages[-1].role == "assistant":
        raise HTTPException(status_code=422, detail="last message must be from user")

    # The prompt constraint: "Max 8 turns per conversation (user + assistant combined)."
    # If len(messages) >= 8, force recommendation.
    max_turns_reached = len(messages) >= 8

    async def _process_chat():
        # 1. Refuse if out of scope
        if await is_out_of_scope(messages[-1].content):
            return ChatResponse(
                reply="I can only help with SHL assessment selection.",
                recommendations=[],
                end_of_conversation=False
            )

        # 2. Synthesize search query from full conversation
        search_query = synthesize_query(messages)

        # 3. Retrieve relevant catalog entries
        catalog_hits = index.search(search_query, top_k=15)

        # 4. Build prompt + call LLM
        response = await call_agent(messages, catalog_hits, max_turns_reached=max_turns_reached)

        # 5. Validate + return
        return validate_and_build_response(response, index)

    try:
        # Enforce 28s timeout as safety margin for 30s limit
        return await asyncio.wait_for(_process_chat(), timeout=28.0)
    except asyncio.TimeoutError:
        return ChatResponse(
            reply="Request timed out. Please try again.",
            recommendations=[],
            end_of_conversation=False
        )
