"""POST /v1/documents (upload 1..5), GET /v1/documents, DELETE /v1/documents/{id} (cascade),
GET /v1/documents/{id}/file, GET /v1/passages/{sentence_id}?context=1  -- all owner-scoped."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import Response

from app.auth import Principal, current_principal
from app.domain import citation_dict
from app.schemas import DocumentInfo, Passage
from app.dependencies import Services, get_services
from app.ratelimit import limit_upload
from app.agent.ingestion import IngestionError

router = APIRouter(prefix="/v1", tags=["documents"])

_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}
_MAX_UPLOAD_BATCH = 5


@router.post("/documents", response_model=list[DocumentInfo], status_code=status.HTTP_201_CREATED)
async def upload_documents(
    files: list[UploadFile] = File(...),
    conversation_id: str = Form(...),
    principal: Principal = Depends(current_principal),
    services: Services = Depends(get_services),
    _rl: None = Depends(limit_upload),
) -> list[dict]:
    if not 1 <= len(files) <= _MAX_UPLOAD_BATCH:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"upload 1..{_MAX_UPLOAD_BATCH} files at a time")
    if not conversation_id.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "conversation_id is required")
    if services.conversations.get(principal.owner_key, conversation_id.strip()) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
    results: list[dict] = []
    for upload in files:
        # Read at most max+1 bytes: an oversized upload is rejected without buffering it all.
        content = await upload.read(services.settings.max_file_bytes + 1)
        try:
            info = await services.ingestion.ingest(
                principal.owner_key, conversation_id.strip(), upload.filename or "document", content
            )
        except IngestionError as exc:
            code = (
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                if "exceeds" in str(exc)
                else status.HTTP_422_UNPROCESSABLE_ENTITY
            )
            raise HTTPException(code, str(exc)) from exc
        results.append(info)
    return results


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents(
    principal: Principal = Depends(current_principal),
    services: Services = Depends(get_services),
) -> list[dict]:
    return services.documents.list(principal.owner_key)


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    principal: Principal = Depends(current_principal),
    services: Services = Depends(get_services),
) -> dict:
    owner = principal.owner_key
    if services.documents.get(owner, document_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    services.sentences.delete_by_document(owner, document_id)
    services.chunks.delete_by_document(owner, document_id)
    services.vectors.delete_document(owner, document_id)
    services.files.delete_document(owner, document_id)
    services.documents.delete(owner, document_id)
    services.bm25.invalidate(owner)
    services.conversations.mark_document_deleted(owner, document_id)
    return {"deleted": document_id}


@router.get("/documents/{document_id}/file")
def get_document_file(
    document_id: str,
    principal: Principal = Depends(current_principal),
    services: Services = Depends(get_services),
) -> Response:
    doc = services.documents.get(principal.owner_key, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "file not found")
    try:
        content = services.files.read(principal.owner_key, document_id)
    except FileNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "file not found")
    ext = os.path.splitext(doc["name"])[1].lower()
    return Response(
        content=content,
        media_type=_MEDIA_TYPES.get(ext, "application/octet-stream"),
        headers={"Content-Disposition": f'attachment; filename="{doc["name"]}"'},
    )


@router.get("/passages/{sentence_id}", response_model=Passage)
def get_passage(
    sentence_id: str,
    context: int = Query(1, ge=0, le=5),
    principal: Principal = Depends(current_principal),
    services: Services = Depends(get_services),
) -> dict:
    owner = principal.owner_key
    sentence = services.sentences.get(owner, sentence_id)
    if sentence is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "passage not found")
    doc = services.documents.get(owner, sentence["document_id"])
    neighbours = services.sentences.neighbors(owner, sentence["document_id"], sentence["idx"], context)
    return {
        "citation": citation_dict(
            n=0,
            sentence_id=sentence["id"],
            document_id=sentence["document_id"],
            document_name=doc["name"] if doc else "",
            text=sentence["text"],
            page=sentence.get("page"),
            heading=sentence.get("heading"),
            bbox=sentence.get("bbox"),
            char_start=sentence.get("char_start"),
            char_end=sentence.get("char_end"),
        ),
        "context_before": [n["text"] for n in neighbours if n["idx"] < sentence["idx"]],
        "context_after": [n["text"] for n in neighbours if n["idx"] > sentence["idx"]],
    }


