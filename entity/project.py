from pydantic import BaseModel
from typing import Optional
class Project(BaseModel):
    uuid:  Optional[str] = None
    name: Optional[str] = None
