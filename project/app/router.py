# Top-level API router: groups endpoint routers under URL prefixes + OpenAPI tags.
# main.py includes this router; endpoints live one level down in endpoint.py.

from fastapi import APIRouter
from endpoint import router as endpoint_router

router = APIRouter()
# prefix=/events → POST /events/; tags=["events"] groups them in /docs.
router.include_router(endpoint_router, prefix="/events", tags=["events"])
