from jose import JWTError, jwt
from fastapi import Header, HTTPException, Depends

SECRET_KEY = "your-secret-key"  # Should match the one in routes.py
ALGORITHM = "HS256"

def get_user_from_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token format.")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # Return all user information from the token
        return {
            "email": payload.get("email") or payload.get("sub"),  # Support both email and sub
            "user_type": payload["user_type"],
            "name": payload.get("name")
        }
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")