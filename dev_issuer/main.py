"""Local stand-in for Microsoft Entra ID (AUTH_MODE=dev). DEMO ONLY - never deploy this.

Generates an RSA key at start-up, serves its public key at /jwks.json and mints RS256 JWTs at
POST /token. The API validates these tokens with exactly the same code path it uses against
a real tenant (signature, issuer, audience, expiry, scope); only the JWKS URL differs."""
import os
import time

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from jwt.algorithms import RSAAlgorithm

ISSUER = os.getenv("DEV_ISSUER", "https://dev-issuer.local")
AUDIENCE = os.getenv("DEV_AUDIENCE", "doctalk-dev")
TENANT = "dev-tenant"
KID = "dev-key-1"

_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_jwk = {**__import__("json").loads(RSAAlgorithm.to_jwk(_key.public_key())), "kid": KID, "use": "sig", "alg": "RS256"}

app = FastAPI(title="DocTalk dev issuer (demo only)")


@app.get("/jwks.json")
def jwks() -> dict:
    return {"keys": [_jwk]}


@app.post("/token")
def token(user: str = "alice", admin: bool = False, ttl_s: int = 3600) -> dict:
    now = int(time.time())
    claims = {
        "iss": ISSUER, "aud": AUDIENCE, "iat": now, "exp": now + ttl_s,
        "tid": TENANT, "oid": f"user-{user}", "scp": "access_as_user",
        "roles": ["DocTalk.Admin"] if admin else [],
    }
    return {"access_token": jwt.encode(claims, _key, algorithm="RS256", headers={"kid": KID}), "token_type": "bearer"}
