"""FastAPI entrypoint: create the app, register routers, run the container for its lifetime."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.container import build_container
from app.routers import admin, conversations, documents

logger = logging.getLogger("doctalk")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)

    services, client = build_container()
    app.state.services = services
    app.state.graph = services.graph

    logger.info(
        "DocTalk started (auth_mode=%s, embeddings=%s)",
        services.settings.auth_mode,
        services.settings.embeddings_configured,
    )
    try:
        yield
    finally:
        client.close()


app = FastAPI(title="DocTalk API", version="0.1.0", lifespan=lifespan)

app.include_router(admin.router)
app.include_router(documents.router)
app.include_router(conversations.router)

