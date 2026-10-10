"""GET /healthz, GET /v1/admin/stats (aggregates only; admin role). Raw log access: admin only."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth import Principal, require_admin
from app.observability import compute_stats
from app.dependencies import Services, get_services

router = APIRouter(tags=["admin"])


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/v1/admin/stats")
def stats(
    principal: Principal = Depends(require_admin),
    services: Services = Depends(get_services),
) -> dict:
    """Aggregates only (no raw text): latency, cost, refusal rate, judge-usage rate."""
    return compute_stats(services.logs)

