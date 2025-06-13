from pydantic import BaseModel, EmailStr

class LoginRequest(BaseModel):
    username: str
    password: str

class User(BaseModel):
    username: str
    password: str
    email: EmailStr

class UserToken(BaseModel):
    username: str
    token: str