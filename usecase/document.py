from repository.document import DocumentRepository
from repository.storage import StorageRepository
from repository.ollama import OllamaAdapter
from entity.document import Document, DocumentDb, Chat
from flask import current_app
from typing import List, Iterator
from static.enum import FileType

class DocumentUsecase:
    def __init__(self, document_repository: DocumentRepository, storage_repository: StorageRepository, ollama_adapter: OllamaAdapter):
        self.document_repository = document_repository
        self.storage_repository = storage_repository
        self.ollama_adapter = ollama_adapter
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
    
    async def document_vectorization(self, document: DocumentDb):
        document_from_db = self.document_repository.get_document(document=document)
        if document_from_db.is_processed:
            return
        self.app.logger.info("document from db name {}, type {}".format(document_from_db.document_name, document_from_db.document_type))
        if document_from_db.document_type == FileType.PDF_DOCUMENT.value:
            langchain_document = await self.storage_repository.load_pdf_document_with_langchain(document=document_from_db)
            self.app.logger.info("pdf loaded with langchain, length {}".format(langchain_document.__len__()) )
            vectorized_ids = self.document_repository.add_documents_to_vector_store(langchain_document)
            self.app.logger.info("vectorized id length {}".format(vectorized_ids.__len__()))
            document_from_db.is_processed = True
            self.document_repository.update_document(document=document_from_db)
            return
        
    def chat_generation(self, chat: Chat) -> str:
        similiar_documents =  self.document_repository.find_relevant_document(chat=chat)
        return self.ollama_adapter.generate_chat_response(chat=chat, similiar_documents=similiar_documents)
    
    def stream_chat_generation(self, chat: Chat) -> Iterator[str]:
        similiar_documents = self.document_repository.find_relevant_document(chat=chat)
        return self.ollama_adapter.generate_streamable_chat_response(chat=chat, similiar_documents=similiar_documents)

    async def summarize_document(self, document: Document):
        self.storage_repository.store_document(document=document)
        langchain_document = await self.storage_repository.load_pdf_document_with_langchain(DocumentDb(document_name=document.file.filename))
        vectorized_ids = self.document_repository.add_documents_to_memory_vector_store(langchain_document)
        self.app.logger.info("vectorized id length {}".format(vectorized_ids.__len__()))
        chat = Chat(chat="tolong ringkas dokumen ini", is_stream=True)
        relevant_documents = self.document_repository.find_relevant_document_from_memory(chat=chat)
        response = self.ollama_adapter.generate_streamable_chat_response(chat=chat, similiar_documents=relevant_documents)
        return response
        