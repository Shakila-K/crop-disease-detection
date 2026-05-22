import logging
import os
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# Same as Govi mobile app backend JWT secret.
# In production, pass this as an environment variable.
JWT_SECRET = os.getenv("GOVI_JWT_SECRET", "default-dev-secret-key-change-in-prod")
JWT_ALGORITHM = "HS256"
DISABLE_JWT_VALIDATION = os.getenv("DISABLE_JWT_VALIDATION", "False").lower() in ("true", "1", "t")

security = HTTPBearer(auto_error=not DISABLE_JWT_VALIDATION)

def verify_jwt(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)) -> dict:
    """
    Verify JWT token from Authorization header.
    Expects Bearer token.
    """
    if DISABLE_JWT_VALIDATION:
        return {"sub": "test-user-id"}

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authenticated",
        )

    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Expired JWT token used")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token used: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(payload: dict = Depends(verify_jwt)) -> str:
    """
    Extracts the user ID from the verified JWT payload.
    Assumes standard 'sub' (subject) or 'user_id' claim.
    """
    user_id = payload.get("sub") or payload.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not determine user from token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id
