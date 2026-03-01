"""MedID search (Supabase)."""
from fastapi import APIRouter
from backend.services.database.medid_repo import search_medid

router = APIRouter(prefix="/medid", tags=["medid"])


@router.get("/search")
async def search_medid_endpoint(q: str):
    results = search_medid(q)
    return {"results": results}
