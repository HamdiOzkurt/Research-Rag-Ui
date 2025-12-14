"""
Clerk JWT verification for FastAPI.

Env vars (recommended):
- CLERK_JWT_ISSUER: e.g. "https://<your-clerk-issuer>"
  If set, JWKS URL defaults to: {CLERK_JWT_ISSUER}/.well-known/jwks.json

Optional:
- CLERK_JWKS_URL: override JWKS URL
- CLERK_JWT_AUDIENCE: expected aud (string)
- CLERK_JWT_PUBLIC_KEY: PEM public key (if you don't want JWKS)
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request


def _bearer_token(req: Request) -> Optional[str]:
    auth = req.headers.get("authorization") or req.headers.get("Authorization")
    if not auth:
        return None
    parts = auth.split(" ", 1)
    if len(parts) != 2:
        return None
    scheme, token = parts[0].strip().lower(), parts[1].strip()
    if scheme != "bearer" or not token:
        return None
    return token


@lru_cache(maxsize=1)
def _jwks_client() -> "jwt.PyJWKClient":
    issuer = os.getenv("CLERK_JWT_ISSUER", "").rstrip("/")
    jwks_url = os.getenv("CLERK_JWKS_URL", "").strip()
    if not jwks_url:
        if not issuer:
            raise RuntimeError(
                "Missing Clerk JWT config. Set CLERK_JWT_ISSUER or CLERK_JWKS_URL (or CLERK_JWT_PUBLIC_KEY)."
            )
        jwks_url = f"{issuer}/.well-known/jwks.json"
    return jwt.PyJWKClient(jwks_url)


def verify_clerk_jwt(token: str) -> dict:
    public_key_pem = os.getenv("CLERK_JWT_PUBLIC_KEY", "").strip()
    issuer = os.getenv("CLERK_JWT_ISSUER", "").rstrip("/") or None
    audience = os.getenv("CLERK_JWT_AUDIENCE", "").strip() or None

    options = {
        "verify_signature": True,
        "verify_exp": True,
        # only verify issuer/audience if provided
        "verify_iss": bool(issuer),
        "verify_aud": bool(audience),
    }

    try:
        if public_key_pem:
            return jwt.decode(
                token,
                public_key_pem,
                algorithms=["RS256"],
                issuer=issuer,
                audience=audience,
                options=options,
            )

        signing_key = _jwks_client().get_signing_key_from_jwt(token).key
        return jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            issuer=issuer,
            audience=audience,
            options=options,
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid auth token: {e}")


def get_current_user_id(req: Request) -> str:
    token = _bearer_token(req)
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <token>")
    claims = verify_clerk_jwt(token)
    user_id = claims.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing 'sub' claim")
    return str(user_id)


def require_user_id(user_id: str = Depends(get_current_user_id)) -> str:
    return user_id


