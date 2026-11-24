from __future__ import annotations

import hmac

from fastapi import HTTPException, Request

from .settings import Settings


def require_access(request: Request, settings: Settings) -> None:
    expected = settings.section("app")["access_token"]
    if not expected:
        return
    authorization = request.headers.get("authorization", "")
    scheme, _, supplied = authorization.partition(" ")
    if scheme.lower() != "bearer" or not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="A valid bearer token is required",
                            headers={"WWW-Authenticate": "Bearer"})