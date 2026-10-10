"""Persistence adapters - the only code that talks to MongoDB and Qdrant.

Every query in `storage` is owner-scoped, so tenant isolation lives in exactly one place.
"""
