"""Basic API key auth (stub for now)."""
from fastapi import APIRouter, Header, HTTPException
from backend.config.settings import settings

router = APIRouter(prefix="/auth", tags=["auth"])


def verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key or x_api_key != settings.API_KEY:
        raise HTTPException(401, "Invalid or missing API key")
    return x_api_key


@router.get("/check")
async def auth_check():
    return {"status": "ok"}
