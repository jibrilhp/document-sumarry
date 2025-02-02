from usecase.document import DocumentUsecase
from usecase.project import ProjectUsecase
from usecase.conversation import ConversationUsecase
from entity.document import Document, DocumentDb, DocumentRequest
from entity.project import Project
from entity.conversation import Conversation, ConversationState
from error.error import FileConflictDb, DatabaseError, ResourceNotFound
from fastapi import Header, status, UploadFile, APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse, Response
from typing import Annotated, List
import logging

class Routes:
    def __init__(self, app: APIRouter, document_usecase: DocumentUsecase, project_usecase: ProjectUsecase, conversation_usecase: ConversationUsecase):
        self.document_usecase = document_usecase
        self.project_usecase = project_usecase
        self.conversation_usecase = conversation_usecase
        self.logger = logging.getLogger(__name__)
        self.app = app
        self.setup_router()

    def setup_router(self):
        @self.app.get("/v1/documents")
        async def get_documents(tenant_id: Annotated[str | None, Header()])->List[DocumentDb]:
            try:
                documentDb = DocumentDb()
                documentDb.set_multinancy_attr(project_uuid="", tenant_id=tenant_id)
                documents = self.document_usecase.get_document(documentDb)
                if documents.__len__() == 0:
                    self.logger.info("document not found")
                    raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
                listOfDocs: List[DocumentDb] = list()
                for document in documents:
                    listOfDocs.append(document.__dict__)
                return listOfDocs
            except HTTPException as e:
                raise e
            except Exception as e:
                self.logger.error(str(e))
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "please try again later")

        @self.app.delete("v1/documents")
        async def delete_document(tenant_id: Annotated[str | None, Header()], document_req: DocumentRequest):
            try:
                document = DocumentDb(uuid=document_req.uuid)
                document.set_multinancy_attr(project_uuid="", tenant_id=tenant_id)
                deleted_document = self.document_usecase.delete_document(document=document)
                if deleted_document.uuid == "":
                    self.logger.info("document not found")
                    raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
                return Response(status_code=status.HTTP_200_OK)
            except ResourceNotFound as e:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
            except DatabaseError as e:
                self.logger.error(str(e))
                raise HTTPException(status.HTTP_404_NOT_FOUND, "please try again later")
            except Exception as e:
                self.logger.error(str(e))
                raise HTTPException(status.HTTP_404_NOT_FOUND, "please try again later")

        @self.app.post("/v1/projects", status_code=status.HTTP_201_CREATED)
        async def create_project(project: Project):
            try:                
                created_project = self.project_usecase.create_project(project=project)
                return created_project
            except FileConflictDb as e:
                self.logger.error(str(e))
                raise HTTPException(status.HTTP_409_CONFLICT, "file already exist in database")
            except Exception as e:
                self.logger.error(str(e))
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "please try again later")

        @self.app.get("/v1/projects")
        async def get_projects():
            try:
                projects = self.project_usecase.get_projects()
                if projects.__len__() == 0:
                    raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
                listOfProject: List[Project] = list()
                for project in projects:
                    listOfProject.append(project)
                return listOfProject
            except HTTPException as e:
                raise e
            except Exception as e:
                self.logger.error(str(e))
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "please try again later")

        @self.app.delete("/v1/projects")
        def delete_project(project: Project):
            try:
                self.project_usecase.delete_project(project=project)
                return Response(status_code=status.HTTP_200_OK)
            except Exception as e:
                self.logger.error(str(e))
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "please try again later")

        @self.app.post("/v1/conversation")
        async def create_chat_session(
            tenant_id: Annotated[str | None, Header()],
            project_uuid: Annotated[str | None, Form()],
            message: Annotated[str | None, Form()],
            conversation_uuid: Annotated[str | None, Form()],
            file: UploadFile | None = None):
            try:
                conversation = Conversation(
                    project_id=project_uuid,
                    conversation_uuid=conversation_uuid,
                    message = message,
                    tenant_id=tenant_id
                )
                if file is not None:
                    document = Document(file=file)
                    document.set_multinancy_attr(project_uuid=project_uuid, tenant_id=tenant_id)
                    document_db = self.document_usecase.store_document(document=document)
                    langchain_document = await self.document_usecase.document_vectorization(document=document_db)
                    conversation.document_from_user = langchain_document
                response_stream = self.conversation_usecase.chat_with_agent(conversation=conversation)
                return JSONResponse({"message": response_stream})
            except Exception as e:
                self.logger.error(str(e))
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "please try again later")

        @self.app.get("/v1/conversation/{conversation_uuid}")
        async def chat_history(
            tenant_id: Annotated[str | None, Header()],
            conversation_uuid: str
        ) -> List[ConversationState]:
            try:
                conversation = Conversation(tenant_id=tenant_id, conversation_uuid=conversation_uuid)
                return self.conversation_usecase.get_chat_history(conversation)
            except Exception as e:
                self.logger.error(str(e))
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "please try again later")
            
        @self.app.get("/")
        def index():
            return "<p>welcome to document summarisation tools</p>"