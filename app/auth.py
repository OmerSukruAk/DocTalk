"""Authentication: Entra ID JWT validation, plus the dev-issuer variant.

Identity comes ONLY from the validated token (owner_key = "tid:oid"). AUTH_MODE=dev
changes the JWKS URL / issuer / audience and nothing else, so the code path exercised
by the demo and by tests is the same one used against a real tenant."""

from __future__ import annotations
from dataclasses import dataclass
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from app.config import Settings, get_settings

# Declared as a security scheme so FastAPI's OpenAPI shows the "Authorize" button
# (and a padlock on each endpoint) in the Swagger UI /docs.
bearer_scheme = HTTPBearer(auto_error=False)


# --- Entra ID validation ---

@dataclass
class Principal:
    owner_key: str
    roles: list[str]


# JWKS clients are cached per URL; PyJWKClient itself caches the fetched key set.
_jwks_clients: dict[str, PyJWKClient] = {}


def _signing_key(jwks_url: str, token: str):
    client = _jwks_clients.get(jwks_url)
    if client is None:
        client = PyJWKClient(jwks_url, cache_keys=True, lifespan=3600)
        _jwks_clients[jwks_url] = client
    return client.get_signing_key_from_jwt(token).key


def _endpoints(settings: Settings) -> tuple[str, str, str]:
    """(jwks_url, issuer, audience) — the only thing that differs between Entra and dev mode."""
    if settings.auth_mode == "dev":
        return settings.dev_issuer_jwks_url, settings.dev_issuer, settings.dev_audience
    tenant = settings.entra_tenant_id
    return (
        f"https://login.microsoftonline.com/{tenant}/discovery/v2.0/keys",
        f"https://login.microsoftonline.com/{tenant}/v2.0",
        settings.entra_audience,
    )


def decode_token(settings: Settings, token: str) -> dict:
    jwks_url, issuer, audience = _endpoints(settings)
    key = _signing_key(jwks_url, token)
    return jwt.decode(
        token,
        key,
        algorithms=["RS256"],
        audience=audience or None,
        issuer=issuer,
        options={"require": ["exp", "iss", "aud"]},
    )


def principal_from_claims(settings: Settings, claims: dict) -> Principal:
    tid = str(claims.get("tid", "")).strip()
    oid = str(claims.get("oid") or claims.get("sub") or "").strip()
    if not tid or not oid:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token missing tid/oid claims")

    scopes = str(claims.get("scp", "")).split()
    roles = list(claims.get("roles", []) or [])
    if settings.entra_required_scope and settings.entra_required_scope not in scopes and roles == []:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "missing required scope")
    return Principal(owner_key=f"{tid}:{oid}", roles=roles)


async def current_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> Principal:
    """FastAPI dependency. The ONLY source of user identity; never trust request fields."""
    services = getattr(request.app.state, "services", None)
    settings: Settings = services.settings if services else get_settings()

    if credentials is None or not credentials.credentials:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    token = credentials.credentials
    try:
        claims = await run_in_threadpool(decode_token, settings, token)
    except HTTPException:
        raise
    except Exception as exc:  # invalid signature / expired / wrong audience
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid token: {exc}") from exc
    return principal_from_claims(settings, claims)


async def require_admin(
    request: Request,
    principal: Principal = Depends(current_principal),
) -> Principal:
    """Dependency for admin-only endpoints (e.g. /v1/admin/stats)."""
    services = getattr(request.app.state, "services", None)
    settings: Settings = services.settings if services else get_settings()
    if settings.admin_role not in principal.roles:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin role required")
    return principal


# --- dev issuer (AUTH_MODE=dev) ---

def build_dev_validator(settings: Settings):
    """Return a callable(token) -> Principal that only differs from Entra by config."""

    def validate(token: str) -> Principal:
        return principal_from_claims(settings, decode_token(settings, token))

    return validate


__all__ = ["Principal", "current_principal", "decode_token", "build_dev_validator"]
