"""API key validation middleware (optional)."""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from backend.config.settings import settings


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/v1/") and request.method != "OPTIONS":
            key = request.headers.get("X-API-Key")
            if not key or key != settings.api_key:
                from starlette.responses import JSONResponse
                return JSONResponse({"detail": "Invalid API key"}, status_code=401)
        return await call_next(request)
