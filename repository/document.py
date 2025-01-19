from repository.data_store import PostgresAdapter, PGVectorAdapter, InMemoryVector
from entity.document import Document, DocumentDb, Chat
from time import time
from flask import current_app
from uuid import uuid5, NAMESPACE_X500
from psycopg2.errors import UniqueViolation
from error.error import FileConflictDb, DatabaseError, ResourceNotFound
from typing import List
from langchain_core.documents import Document as LangchaincoreDocument


class DocumentRepository:
    def __init__(self, db: PostgresAdapter, pgvector: PGVectorAdapter, inmemory_vector: InMemoryVector):
        self.app = current_app
        self.cursor = db.get_cursor()
        self.connection = db.get_connection()
        self.pg_vector_store = pgvector
        self.inmmeory_vector_store = inmemory_vector

    def get_documents(self) -> List[DocumentDb]:
        sql = "select uuid, document_name, is_processed, document_type, created_at, updated_at from documents"
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        documents: List[DocumentDb] = list()
        for row in results:
            uuid, document_name, is_processed, document_type, created_at, updated_at = row
            document = DocumentDb(
                uuid=uuid, document_name=document_name, is_processed=is_processed,
                document_type=document_type, created_at=created_at, updated_at=updated_at
            )
            documents.append(document)

        return documents
    
    def get_document(self, document: DocumentDb)-> DocumentDb:
        sql = "select uuid, document_name, is_processed, document_type, created_at, updated_at from documents where uuid = %s limit 1"
        data = (document.uuid,)
        self.cursor.execute(sql, data)
        results = self.cursor.fetchall()
        if results.__len__() == 0:
            e = ResourceNotFound("document not found")
            self.app.logger.error(str(e))
            raise ResourceNotFound("document not found")
        for row in results:
            uuid, document_name, is_processed, document_type, created_at, updated_at = row
            document = DocumentDb(
                uuid=uuid, document_name=document_name, is_processed=is_processed,
                document_type=document_type, created_at=created_at, updated_at=updated_at
            )
            return document

    def store_document(self, document: Document):
        sql = "INSERT INTO documents(uuid, document_name, is_processed, document_type, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s);"
        data = (
           uuid5(NAMESPACE_X500, document.file.filename).__str__() , document.file.filename, False, document.file_type, time(), time()
        )
        try:
            self.cursor.execute(query=sql, vars=data)
            self.connection.commit()
            self.app.logger.info("document {} is stored into db".format(document.file.filename))
        except UniqueViolation:
            self.connection.rollback()
            raise FileConflictDb("{} already exist in database".format(document.file.filename))
        except Exception:
            self.connection.rollback()
            raise DatabaseError("please try again later")
        
    def update_document(self, document: DocumentDb):
        sql = "update documents set is_processed = %s, updated_at = %s where uuid = %s"
        data = (document.is_processed, time(), document.uuid)
        try:
            self.cursor.execute(query=sql, vars=data)
            self.connection.commit()
        except Exception as e:
            self.app.logger.error(str(e))
            self.connection.rollback()
            raise DatabaseError("please try again later")
        
    def delete_document(self, document: DocumentDb):
        sql = "DELETE FROM documents WHERE uuid = %s"
        data = (document.uuid,)
        self.app.logger.info("deleting document with uuid: {}".format(document.uuid))
        try:
            self.cursor.execute(query=sql, vars=data)
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            self.app.logger.error(e)
            raise DatabaseError("please try again later")
        
    def add_documents_to_vector_store(self, documents: List[LangchaincoreDocument]) -> List[str]:
        try:
            ids = self.pg_vector_store.vector_store.add_documents(documents=documents)
            return ids
        except ValueError:
            return list<str>()
        
    def find_relevant_document(self, chat: Chat):
        query = chat.chat
        similiar_documents = self.pg_vector_store.vector_store.similarity_search(query=query, k=3)
        return similiar_documents
    
    def add_documents_to_memory_vector_store(self, documents: List[LangchaincoreDocument]) -> List[str]:
        try:
            ids = self. inmmeory_vector_store.in_memory_vector_store.add_documents(documents=documents)
            return ids
        except ValueError:
            return list<str>()
        
    def find_relevant_document_from_memory(self, chat: Chat):
        query = chat.chat
        similiar_documents = self.inmmeory_vector_store.in_memory_vector_store.similarity_search(query=query, k=3)
        return similiar_documents
    