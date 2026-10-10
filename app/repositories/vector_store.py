"""Qdrant wrapper: one collection, payload = {chunk_id, document_id, owner_key}.

Records the embed model + dimension in the collection metadata; refuses to mix models.
Chunks (parent passages) are the embedded unit; citations still resolve to sentences."""

from __future__ import annotations
from qdrant_client import QdrantClient, models as qm
from app.config import Settings


# --- vector store (Qdrant) ---

class EmbedModelMismatch(RuntimeError):
    pass


class VectorStore:
    def __init__(self, settings: Settings, collection: str = "chunks"):
        self._settings = settings
        self._collection = collection
        self._meta_collection = f"{collection}__meta"
        self._client = QdrantClient(url=settings.qdrant_url)

    # -- lifecycle --------------------------------------------------------
    def ensure_collection(self, embed_model: str, dim: int) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            meta = self._read_meta()
            if meta and (meta.get("embed_model") != embed_model or int(meta.get("dim", 0)) != dim):
                raise EmbedModelMismatch(
                    f"Collection '{self._collection}' was built with {meta}; refusing to mix "
                    f"{embed_model}/{dim}. Re-index or use a new collection."
                )
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="owner_key",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="document_id",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
        self._write_meta(embed_model, dim)

    def _read_meta(self) -> dict | None:
        try:
            points = self._client.retrieve(self._meta_collection, ids=[0], with_payload=True)
        except Exception:
            return None
        return points[0].payload if points else None

    def _write_meta(self, embed_model: str, dim: int) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._meta_collection not in existing:
            self._client.create_collection(
                collection_name=self._meta_collection,
                vectors_config=qm.VectorParams(size=1, distance=qm.Distance.DOT),
            )
        self._client.upsert(
            collection_name=self._meta_collection,
            points=[
                qm.PointStruct(
                    id=0,
                    vector=[0.0],
                    payload={"embed_model": embed_model, "dim": dim},
                )
            ],
        )

    # -- data -------------------------------------------------------------
    def upsert(self, owner_key: str, points: list[tuple[str, str, list[float]]]) -> None:
        """points: list of (chunk_id, document_id, vector)."""
        structs = [
            qm.PointStruct(
                id=_point_id(owner_key, cid),
                vector=vec,
                payload={"chunk_id": cid, "document_id": did, "owner_key": owner_key},
            )
            for cid, did, vec in points
        ]
        if structs:
            self._client.upsert(collection_name=self._collection, points=structs, wait=True)

    def search(self, owner_key: str, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        flt = qm.Filter(must=[qm.FieldCondition(key="owner_key", match=qm.MatchValue(value=owner_key))])
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            query_filter=flt,
            with_payload=True,
        )
        return [(p.payload["chunk_id"], float(p.score)) for p in response.points]

    def delete_document(self, owner_key: str, document_id: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(key="owner_key", match=qm.MatchValue(value=owner_key)),
                        qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id)),
                    ]
                )
            ),
            wait=True,
        )

    def delete_owner(self, owner_key: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="owner_key", match=qm.MatchValue(value=owner_key))]
                )
            ),
            wait=True,
        )


def _point_id(owner_key: str, chunk_id: str) -> int:
    import uuid

    return uuid.uuid5(uuid.NAMESPACE_URL, f"{owner_key}:{chunk_id}").int & ((1 << 63) - 1)
