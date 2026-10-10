"""Per-user sliding-window rate limiting (in-process).

Protects the paid LLM/embedding budget from a single noisy user. It is per replica by design
(no extra infrastructure for the case study); in production put Azure API Management or a
Redis-backed limiter in front, see README -> Limitations."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request, status

from app.auth import Principal, current_principal


class SlidingWindowLimiter:
    def __init__(self, window_s: float = 60.0):
        self._window = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, limit: int, now: float | None = None) -> float:
        """Record a hit. Returns 0 if allowed, else the seconds until the next slot frees up."""
        if limit <= 0:
            return 0.0
        now = time.monotonic() if now is None else now
        hits = self._hits[key]
        while hits and now - hits[0] >= self._window:
            hits.popleft()
        if len(hits) >= limit:
            return max(0.0, self._window - (now - hits[0]))
        hits.append(now)
        return 0.0


_limiter = SlidingWindowLimiter()


def rate_limit(bucket: str, setting_name: str):
    async def dependency(request: Request, principal: Principal = Depends(current_principal)) -> None:
        services = getattr(request.app.state, "services", None)
        limit = int(getattr(services.settings, setting_name, 0)) if services else 0
        retry_after = _limiter.check(f"{bucket}:{principal.owner_key}", limit)
        if retry_after:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "rate limit exceeded, slow down",
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

    return dependency


limit_ask = rate_limit("ask", "rate_limit_ask_per_min")
limit_upload = rate_limit("upload", "rate_limit_upload_per_min")
