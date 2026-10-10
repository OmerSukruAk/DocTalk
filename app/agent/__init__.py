"""Business logic: the RAG pipeline, end to end.

ingestion -> retrieval -> generation -> verification, orchestrated by the LangGraph in
`workflow`. Nothing here knows about HTTP, or about how its output is persisted.
"""
