"""POST/GET/DELETE /v1/conversations, GET /v1/conversations/{id},
POST /v1/conversations/{id}/messages -> AskResponse (runs the LangGraph workflow)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.auth import Principal, current_principal
from app.schemas import AskRequest, AskResponse
from app.dependencies import Services, get_graph, get_services
from app.ratelimit import limit_ask
from app.agent.workflow import answer_question

router = APIRouter(prefix="/v1", tags=["conversations"])


@router.post("/conversations", status_code=status.HTTP_201_CREATED)
def create_conversation(
    principal: Principal = Depends(current_principal),
    services: Services = Depends(get_services),
) -> dict:
    conversation_id = services.conversations.create(principal.owner_key)
    return services.conversations.get(principal.owner_key, conversation_id)


@router.get("/conversations")
def list_conversations(
    principal: Principal = Depends(current_principal),
    services: Services = Depends(get_services),
) -> list[dict]:
    return services.conversations.list(principal.owner_key)


@router.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str,
    principal: Principal = Depends(current_principal),
    services: Services = Depends(get_services),
) -> dict:
    owner = principal.owner_key
    conversation = services.conversations.get(owner, conversation_id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
    return {
        "conversation": conversation,
        "messages": services.conversations.list_messages(owner, conversation_id),
    }


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    principal: Principal = Depends(current_principal),
    services: Services = Depends(get_services),
) -> dict:
    if not services.conversations.delete(principal.owner_key, conversation_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
    return {"deleted": conversation_id}


@router.post("/conversations/{conversation_id}/messages", response_model=AskResponse)
async def ask(
    conversation_id: str,
    body: AskRequest,
    principal: Principal = Depends(current_principal),
    services: Services = Depends(get_services),
    graph=Depends(get_graph),
    _rl: None = Depends(limit_ask),
) -> dict:
    owner = principal.owner_key
    if await run_in_threadpool(services.conversations.get, owner, conversation_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
    question = body.question.strip()
    if not question:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "question must not be empty")
    return await answer_question(
        services, graph, owner_key=owner, conversation_id=conversation_id, question=question,
        include_public=body.include_public, style=body.style,
    )

