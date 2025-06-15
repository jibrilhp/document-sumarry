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

class UserAccessTokenRequest(BaseModel):
    description: str

class UserAccessTokenResponse(BaseModel):
    api_token: str
    description: str
    message: str
