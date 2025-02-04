from pydantic import BaseModel

class SuccessResponse(BaseModel):
    message: str

class ErrorResponse(BaseModel):
    error: str