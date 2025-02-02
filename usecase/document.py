from repository.document import DocumentRepository
from infra.storage import StorageRepository
from infra.generative_provider import GenerativeAdapter
from entity.document import Document, DocumentDb
from typing import List
from static.enum import FileType
from langchain_core.documents import Document as LangchainDocument
import logging

class DocumentUsecase:
    def __init__(self, document_repository: DocumentRepository, storage_repository: StorageRepository, ollama_adapter: GenerativeAdapter):
        self.document_repository = document_repository
        self.storage_repository = storage_repository
        self.ollama_adapter = ollama_adapter
        self.logger = logging.getLogger(__name__)

    def store_document(self, document: Document):
        document_db_from_file = DocumentDb(document_name=document.file.filename)
        document_db_from_file.set_multinancy_attr(project_uuid=document.project_uuid, tenant_id=document.tenant_id)
        document_db = self.document_repository.get_document_by_name(document_db_from_file)
        if document_db.document_name != "":
            self.logger.info("document already exist: {}".format(document_db.document_name))
            return document_db
        self.logger.info("processing file {}".format( document.file.filename))
        self.storage_repository.store_document(document=document)
        return self.document_repository.store_document(document=document)
    
    def get_document(self, documentDb: DocumentDb) -> List[DocumentDb]:
        documents = self.document_repository.get_documents(documentDb=documentDb)
        return documents
    
    def delete_document(self, document: DocumentDb) -> DocumentDb:
        document_from_db = self.document_repository.get_document(document=document)
        self.document_repository.delete_document(document=document_from_db)
        return document_from_db

    async def document_vectorization(self, document: DocumentDb):
        document_from_db = self.document_repository.get_document(document=document)
        self.logger.info("document from db name {}, type {}".format(document_from_db.document_name, document_from_db.document_type))
        if document_from_db.document_type == FileType.PDF_DOCUMENT.value:
            langchain_document = await self.storage_repository.load_pdf_document_with_langchain(document=document_from_db)
            self.logger.info("pdf loaded with langchain, length {}".format(langchain_document.__len__()) )
            return self.__process_document(document_db=document_from_db, langchain_document=langchain_document)
        if document_from_db.document_type == FileType.IMAGE_DOCUMENT.value:
            langchain_document = self.storage_repository.load_image_with_langchain(image_db=document_from_db)
            self.logger.info("image loaded with langchain, length {}".format(langchain_document.__len__()))
            return self.__process_document(document_db=document_from_db, langchain_document=langchain_document)
        
    def __process_document(self, document_db: DocumentDb, langchain_document: List[Document]):
            if document_db.is_processed:
                return langchain_document
            vectorized_ids = self.document_repository.add_documents_to_vector_store(langchain_document)
            self.logger.info("vectorized id length {}".format(vectorized_ids.__len__()))
            document_db.is_processed = True
            self.document_repository.update_document(document=document_db)
            return langchain_document