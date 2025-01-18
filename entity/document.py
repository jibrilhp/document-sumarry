from werkzeug.datastructures import FileStorage
from static.enum import FileType
from error.error import UnknownFileType

class Document:
    def __init__(self, file: FileStorage):
        self.file: FileStorage = file
        self.is_processed = False
        self.file_type = self.__check_file_type(file=file)

    def __check_file_type(self, file: FileStorage) -> int:
        file_name = file.filename
        if file_name.endswith(".pdf"):
            return FileType.PDF_DOCUMENT.value
        if file_name.endswith(".png") or file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
            return FileType.IMAGE_DOCUMENT.value
        raise UnknownFileType(message="invalid file type", file_type=file_name)
    
class DocumentDb(object):
    def __init__(self, uuid: str="", document_name: str="", is_processed: bool=False, document_type: int=0, created_at: int=0, updated_at: int=0):
        self.uuid = uuid
        self.document_name = document_name
        self.is_processed = is_processed
        self.document_type = document_type
        self.created_at = created_at
        self.updated_at = updated_at
