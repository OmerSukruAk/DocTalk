"""FastAPI entrypoint: create the app, register routers, run the container for its lifetime."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.container import build_container
from app.routers import admin, conversations, documents

logger = logging.getLogger("doctalk")


def _warm_up(services) -> None:
    """Load the optional local models (re-ranker, NLI) once, so the first question is as
    fast as the rest. Importing torch and loading a cross-encoder takes ~20 s on a CPU."""
    for model in (services.reranker, services.nli):
        try:
            model.warm_up()
        except Exception as exc:  # never block startup: the model loads on first use instead
            logger.warning("Model warm-up failed for %s: %s", type(model).__name__, exc)
    logger.info("Model warm-up finished")


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
    # in a background thread: the API accepts requests at once, and a question asked during
    # the warm-up simply waits for the model instead of loading it a second time
    warm_up = asyncio.create_task(asyncio.to_thread(_warm_up, services))
    try:
        yield
    finally:
        await warm_up
        client.close()


app = FastAPI(title="DocTalk API", version="0.1.0", lifespan=lifespan)

app.include_router(admin.router)
app.include_router(documents.router)
app.include_router(conversations.router)

