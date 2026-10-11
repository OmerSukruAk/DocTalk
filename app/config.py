"""Typed settings loaded from environment (.env). One Settings object, injected everywhere."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Mirrors .env.example one-for-one. Extra env vars are ignored so partial
    .env files never crash the app."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # --- LLM (OpenAI-compatible; DeepSeek by default) ---
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_price_in_per_mtok: float = 0.0
    llm_price_out_per_mtok: float = 0.0

    # --- Embeddings (OpenAI or Azure OpenAI; DeepSeek has no embeddings endpoint) ---
    embed_provider: str = "openai"  # openai | azure
    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = "text-embedding-3-small"
    embed_dimensions: int = 1536

    # --- Local models ---
    nli_model: str = ""
    rerank_model: str = ""

    # --- Storage ---
    mongo_uri: str = "mongodb://mongo:27017"
    mongo_db: str = "doctalk"
    qdrant_url: str = "http://qdrant:6333"

    # --- Auth ---
    auth_mode: str = "entra"  # entra | dev
    entra_tenant_id: str = ""
    entra_audience: str = ""
    entra_required_scope: str = "access_as_user"
    admin_role: str = "DocTalk.Admin"
    dev_issuer_jwks_url: str = "http://dev-issuer:9000/jwks.json"
    dev_issuer: str = "https://dev-issuer.local"
    dev_audience: str = "doctalk-dev"

    # --- Behaviour ---
    max_docs_per_conversation: int = 5
    max_file_mb: int = 20
    top_k: int = 8
    chunk_max_tokens: int = 256
    chunk_min_tokens: int = 80  # sections smaller than this are merged into a neighbour
    chunk_overlap_sentences: int = 1  # embedding-only overlap between chunks of one section
    rerank_candidates: int = 24
    context_neighbours: int = 1
    # Relevance gate (runs BEFORE any LLM call; calibrate with eval/run_eval.py):
    gate_min_score: float = 0.0  # legacy floor on the fused RRF score (kept for compatibility)
    gate_min_vector_score: float = 0.25  # best cosine similarity of the query to any chunk
    gate_min_rerank_score: float = 0.05  # cross-encoder probability (used when RERANK_MODEL is set)
    gate_min_lexical_coverage: float = 0.5  # share of the query's content words found in the best chunk
    nli_pass: float = 0.8
    nli_fail: float = 0.2
    max_retries: int = 1
    log_text: bool = False  # privacy by default: do not persist question/answer text in logs
    rate_limit_ask_per_min: int = 20  # per user, per replica; 0 disables
    rate_limit_upload_per_min: int = 10
    log_ttl_days: int = 90

    # --- Derived helpers -------------------------------------------------

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024

    @property
    def embeddings_configured(self) -> bool:
        return bool(self.embed_api_key and self.embed_model)

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_model)

    @property
    def nli_configured(self) -> bool:
        return bool(self.nli_model)

    @property
    def rerank_configured(self) -> bool:
        return bool(self.rerank_model)

    def cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        return (
            tokens_in / 1_000_000 * self.llm_price_in_per_mtok
            + tokens_out / 1_000_000 * self.llm_price_out_per_mtok
        )


@lru_cache
def get_settings() -> Settings:
    from app.secrets import load_key_vault_into_env

    load_key_vault_into_env()  # no-op unless KEY_VAULT_URL is set
    return Settings()
