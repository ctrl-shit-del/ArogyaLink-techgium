"""Per-route rate limiting (stub)."""
# Stub: implement with slowapi or redis-based counter
def rate_limit(requests_per_minute: int = 60):
    def decorator(func):
        return func
    return decorator
