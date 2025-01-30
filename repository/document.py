from infra.data_store import PostgresAdapter, PGVectorAdapter, InMemoryVector
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

    def get_documents(self, documentDb: DocumentDb) -> List[DocumentDb]:
        sql = "select uuid, document_name, is_processed, document_type_id, created_at, updated_at from documents where tenant_id = %s"
        data = (documentDb.tenant_id,)
        self.cursor.execute(sql, vars=data)
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
        sql = "select uuid, document_name, is_processed, document_type_id, created_at, updated_at, tenant_id, projects_uuid from documents where uuid = %s AND tenant_id = %s limit 1"
        data = (document.uuid,document.tenant_id)
        self.app.logger.info("get document with where {} {}".format( document.uuid, document.tenant_id))
        self.cursor.execute(sql, data)
        results = self.cursor.fetchall()
        if results.__len__() == 0:
            raise ResourceNotFound("resource not found")
        for row in results:
            uuid, document_name, is_processed, document_type, created_at, updated_at, tenant_id, project_uuid = row
            document = DocumentDb(
                uuid=uuid, document_name=document_name, is_processed=is_processed,
                document_type=document_type, created_at=created_at, updated_at=updated_at
            )
            document.set_multinancy_attr(project_uuid=project_uuid, tenant_id=tenant_id)
            return document

    def store_document(self, document: Document):
        sql = "INSERT INTO documents(uuid, document_name, is_processed, document_type_id, created_at, updated_at, projects_uuid, tenant_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);"
        uid = uuid5(NAMESPACE_X500, document.file.filename).__str__()
        data = (
           uid , document.file.filename, False, document.file_type, time(), time(), document.project_uuid, document.tenant_id
        )
        try:
            self.cursor.execute(query=sql, vars=data)
            self.connection.commit()
            self.app.logger.info("document {} is stored into db".format(document.file.filename))
            document_db = DocumentDb(uuid=uid, document_name=document.file.filename, document_type=document.file_type)
            document_db.set_multinancy_attr(project_uuid=document.project_uuid, tenant_id=document.tenant_id)
            return document_db
        except UniqueViolation as e:
            self.app.logger.error(str(e))
            self.connection.rollback()
            raise FileConflictDb("{} already exist in database".format(document.file.filename))
        except Exception as e:
            self.app.logger.error(str(e))
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
        sql = "DELETE FROM documents WHERE uuid = %s AND tenant_id = %s"
        data = (document.uuid,document.tenant_id)
        self.app.logger.info("delete document with where {} {}".format(document.uuid, document.tenant_id))
        try:
            self.cursor.execute(query=sql, vars=data)
            self.connection.commit()
        except Exception as e:
            self.app.logger.error(str(e))
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
        self.app.logger.info("find relevant document with fiter tenant_id: {} and project_uuid: {}".format(chat.tenant_id, chat.project_uuid))
        query = chat.chat
        similiar_documents = self.pg_vector_store.vector_store.similarity_search_with_relevance_scores(query=query, filter={"tenant_id":chat.tenant_id, "project_uuid":chat.project_uuid}, score_threshold=0.7)
        sd = list()
        for similiar_document in similiar_documents:
            sd.append(similiar_document[0])
        return sd
    