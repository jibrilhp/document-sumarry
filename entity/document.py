from static.enum import FileType
from error.error import UnknownFileType
from time import time
from pydantic import BaseModel
from fastapi import UploadFile as FileStorage
from typing import List, Dict

class DocumentRequest(BaseModel):
    uuid: str | None
    name: str | None
    
class Document:
    def __init__(self, file: FileStorage):
        self.file: FileStorage = file
        self.is_processed = False
        self.file_type = self.__check_file_type(file=file)
        self.project_uuid = ""
        self.tenant_id = ""

    def __check_file_type(self, file: FileStorage) -> int:
        file_name = file.filename
        if file_name.endswith(".pdf"):
            return FileType.PDF_DOCUMENT.value
        if file_name.endswith(".png") or file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
            return FileType.IMAGE_DOCUMENT.value
        if file_name.endswith(".csv"):
            return FileType.CSV_DOCUMENT.value
        if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            return FileType.EXCEL_DOCUMENT.value
        raise UnknownFileType(message="invalid file type", file_type=file_name)
    
    def set_multinancy_attr(self, project_uuid: str, tenant_id: str):
        self.project_uuid = project_uuid
        self.tenant_id = tenant_id
    
class DocumentDb(BaseModel):
    uuid : str = ""
    document_name : str = ""
    is_processed : bool = False
    document_type : int = 0
    created_at : int = 0
    updated_at : int = 0
    project_uuid : str = ""
    tenant_id : str = ""
        
    def set_multinancy_attr(self, project_uuid: str, tenant_id: str):
        self.project_uuid = project_uuid
        self.tenant_id = tenant_id

class Chat(object):
    def __init__(self, chat: str, is_stream: bool):
        self.timestamp = time()
        self.chat = chat
        self.is_stream = is_stream
        self.project_uuid = ""
        self.tenant_id = ""

    def set_multinancy_attr(self, project_uuid: str, tenant_id: str):
        self.project_uuid = project_uuid
        self.tenant_id = tenant_id


class UploadXAIFile(BaseModel):
    tenant_id: str
    project_uuid: str
    file: FileStorage

class FeatureImpact(BaseModel):
    name: str
    mean_value: float

class ShapSummary(BaseModel):
    file_name: str
    prediction_distribution: Dict[str, int]
    top_positive: List[FeatureImpact]
    top_negative: List[FeatureImpact]

class XAIFileSummary(BaseModel):
    summary: str
    source: str