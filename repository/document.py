from infra.data_store import PostgresAdapter
from entity.document import Document, DocumentDb
from time import time
from uuid import uuid5, NAMESPACE_X500
from psycopg2.errors import UniqueViolation
from error.error import FileConflictDb, DatabaseError, ResourceNotFound
from typing import List
from langchain_core.documents import Document as LangchaincoreDocument
import logging

class DocumentRepository:
    def __init__(self, db: PostgresAdapter):
        self.logger = logging.getLogger(__name__)
        self.db_adapter = db
        self.pg_vector_store = db.get_vector_store()

    def get_documents(self, documentDb: DocumentDb) -> List[DocumentDb]:
        with self.db_adapter.get_connection() as conn:
            self.logger.info("finding document from tenants {}".format(documentDb.tenant_id))
            try:
                sql = "select uuid, document_name, is_processed, document_type_id, created_at, updated_at from documents where tenant_id = %s"
                data = (documentDb.tenant_id,)
                results = conn.cursor().execute(sql, data).fetchall()
                documents: List[DocumentDb] = list()
                for row in results:
                    uuid, document_name, is_processed, document_type, created_at, updated_at = row
                    document = DocumentDb(
                        uuid=uuid, document_name=document_name, is_processed=is_processed,
                        document_type=document_type, created_at=created_at, updated_at=updated_at
                    )
                    documents.append(document)
                return documents
            except Exception as e:
                self.logger.error(str(e))
                conn.rollback()
                raise DatabaseError(e.__str__())
    
    def get_document(self, document: DocumentDb)-> DocumentDb:
        with self.db_adapter.get_connection() as conn:
            try:
                sql = "select uuid, document_name, is_processed, document_type_id, created_at, updated_at, tenant_id, projects_uuid from documents where uuid = %s AND tenant_id = %s limit 1"
                data = (document.uuid,document.tenant_id)
                self.logger.info("get document with where {} {}".format( document.uuid, document.tenant_id))
                results = conn.execute(sql, data).fetchall()
                for row in results:
                    uuid, document_name, is_processed, document_type, created_at, updated_at, tenant_id, project_uuid = row
                    document = DocumentDb(
                        uuid=uuid, document_name=document_name, is_processed=is_processed,
                        document_type=document_type, created_at=created_at, updated_at=updated_at
                    )
                    document.set_multinancy_attr(project_uuid=project_uuid, tenant_id=tenant_id)
                    return document
            except Exception as e:
                self.logger.error(str(e))
                conn.rollback()
                raise DatabaseError(e.__str__())
    
    def get_document_by_name(self, document: DocumentDb)-> DocumentDb:
        with self.db_adapter.get_connection() as conn:
            try:
                sql = "select uuid, document_name, is_processed, document_type_id, created_at, updated_at, tenant_id, projects_uuid from documents where document_name = %s AND tenant_id = %s limit 1"
                data = (document.document_name,document.tenant_id)
                self.logger.info("get document with where {} {}".format( document.uuid, document.tenant_id))
                results = conn.execute(sql, data).fetchall()
                for row in results:
                    uuid, document_name, is_processed, document_type, created_at, updated_at, tenant_id, project_uuid = row
                    document = DocumentDb(
                        uuid=uuid, document_name=document_name, is_processed=is_processed,
                        document_type=document_type, created_at=created_at, updated_at=updated_at
                    )
                    document.set_multinancy_attr(project_uuid=project_uuid, tenant_id=tenant_id)
                    return document
            except Exception as e:
                self.logger.error(str(e))
                conn.rollback()
                raise DatabaseError(e.__str__())

    def store_document(self, document: Document):
        with self.db_adapter.get_connection() as conn:
            try:
                sql = "INSERT INTO documents(uuid, document_name, is_processed, document_type_id, created_at, updated_at, projects_uuid, tenant_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);"
                uid = uuid5(NAMESPACE_X500, document.file.filename).__str__()
                data = (
                uid , document.file.filename, False, document.file_type, time(), time(), document.project_uuid, document.tenant_id
                )
                conn.execute(sql, data)
                conn.commit()
                self.logger.info("document {} is stored into db".format(document.file.filename))
                document_db = DocumentDb(uuid=uid, document_name=document.file.filename, document_type=document.file_type)
                document_db.set_multinancy_attr(project_uuid=document.project_uuid, tenant_id=document.tenant_id)
                return document_db
            except UniqueViolation as e:
                self.logger.error(str(e))
                conn.rollback()
                raise FileConflictDb("{} already exist in database".format(document.file.filename))
            except Exception as e:
                self.logger.error(str(e))
                conn.rollback()
                raise DatabaseError("please try again later")
        
    def update_document(self, document: DocumentDb):
        with self.db_adapter.get_connection() as conn:
            sql = "update documents set is_processed = %s, updated_at = %s where uuid = %s"
            data = (document.is_processed, time(), document.uuid)
            try:
                conn.execute(query=sql, vars=data)
                conn.commit()
            except Exception as e:
                self.logger.error(str(e))
                conn.rollback()
                raise DatabaseError("please try again later")
        
    def delete_document(self, document: DocumentDb):
        with self.db_adapter.get_connection() as conn:
            sql = "DELETE FROM documents WHERE uuid = %s AND tenant_id = %s"
            data = (document.uuid,document.tenant_id)
            self.logger.info("delete document with where {} {}".format(document.uuid, document.tenant_id))
            try:
                conn.execute(query=sql, vars=data)
                conn.commit()
            except Exception as e:
                self.logger.error(str(e))
                conn.rollback()
                raise DatabaseError("please try again later")
        
    def add_documents_to_vector_store(self, documents: List[LangchaincoreDocument]) -> List[str]:
        try:
            ids = self.pg_vector_store.vector_store.add_documents(documents=documents)
            return ids
        except ValueError:
            return list<str>()