from repository.document import DocumentRepository
from repository.storage import StorageRepository
from entity.document import Document, DocumentDb
from flask import current_app
from typing import List

class DocumentUsecase:
    def __init__(self, document_repository: DocumentRepository, storage_repository: StorageRepository):
        self.document_repository = document_repository
        self.storage_repository = storage_repository
        self.app = current_app

    def store_document(self, document: Document):
        self.app.logger.info("processing file {}".format( document.file.filename))
        self.storage_repository.store_document(document=document)
        self.document_repository.store_document(document=document)
        return
    
    def get_document(self) -> List[DocumentDb]:
        documents = self.document_repository.get_documents()
        return documents
    
    def delete_document(self, document: DocumentDb):
        document_from_db = self.document_repository.get_document(document=document)
        self.document_repository.delete_document(document=document_from_db)
        return
