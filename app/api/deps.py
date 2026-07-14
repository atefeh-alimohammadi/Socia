from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2AuthorizationCodeBearer, OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.database.deps import get_db
from app.models.user import User
from app.utils.token import verify_token

oath2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login"
)

def get_current_user(
        token: str = Depends(oath2_scheme),
        db: Session = Depends(get_db) ):
    payload = verify_token(token)
    email = payload.get("sub")

    if email is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.email == email).first()

    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    return user
