"""DocTalk API.

    main.py            ASGI target (uvicorn app.main:app) and lifespan wiring
    container.py       build_container() - the composition root; the only place objects are built
    config.py          Settings, loaded once from the environment
    domain.py          Sentence + citation shape - shared types that cross layer boundaries
    auth.py            who the caller is - Entra ID, or the dev issuer
    schemas.py         the request/response contract
    observability.py   structured logs and metrics
    dependencies.py    the Services container handed to routers via Depends()
    routers/           HTTP edge - authenticate, validate, delegate
    agent/             business logic - the RAG workflow + tools
    repositories/      persistence - MongoDB and Qdrant

Dependencies flow one way: routers -> agent -> repositories.
"""

