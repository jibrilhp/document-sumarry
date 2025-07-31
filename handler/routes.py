from usecase.document import DocumentUsecase
from usecase.project import ProjectUsecase
from usecase.conversation import ConversationUsecase
from usecase.user import UserUsecase
from entity.user import LoginRequest, UserAccessTokenRequest, UserAccessTokenResponse
from entity.document import Document, DocumentDb, DocumentRequest
from entity.project import Project
from entity.conversation import Conversation, ConversationState
from error.error import FileConflictDb, DatabaseError, ResourceNotFound, UnknownFileType, FileTooLarge, UnauthorizedAccess
from fastapi import Header, status, UploadFile, APIRouter, HTTPException, Form, Query, Request, Depends
from fastapi.responses import JSONResponse, Response, StreamingResponse, FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import Annotated, List
import logging
import magic
import jwt
from infra.settings import Settings

class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.__logger = logging.getLogger(__name__)
        self.__settings = settings

    async def dispatch(self, request: Request, call_next):
        token = request.headers.get("Authorization")
        if request.url.path.__contains__("/login") or request.url.path.__contains__("/internal"):
            self.__logger.info(f"path={request.url.path} method={request.method} client_ip={request.client.host}")
            return await call_next(request)
        if token is None:
            return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Not authenticated"})

        try:
            payload = jwt.decode(token.replace("Bearer ", ""), self.__settings.SECRET_KEY, algorithms=self.__settings.JWT_ALGORITHM)
            sub = payload.get("sub")
            if sub is None:
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Not authenticated"})
            
            request.state.user = sub
            response = await call_next(request)
            if hasattr(request.state, "req_token") and hasattr(request.state, "res_token"):
                self.__logger.info(f"user={sub} path={request.url.path} method={request.method} request_token={request.state.req_token}, response_token={request.state.res_token}")
            else:
                self.__logger.info(f"user={sub} path={request.url.path} method={request.method}")
            return response
        except jwt.ExpiredSignatureError:
            return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Token has expired"})
        except jwt.InvalidTokenError:
            return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Invalid token"})
        except Exception as e:
            logging.error(f"Error decoding token: {str(e)}")
            return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Not authenticated"})

class Routes:
    def __init__(
            self, 
            app: APIRouter, 
            document_usecase: DocumentUsecase, 
            project_usecase: ProjectUsecase, 
            conversation_usecase: ConversationUsecase, 
            settings: Settings,
            user_usecase: UserUsecase
            ):
        self.document_usecase = document_usecase
        self.project_usecase = project_usecase
        self.conversation_usecase = conversation_usecase
        self.user_usecase = user_usecase
        self.logger = logging.getLogger(__name__)
        self.app = app
        self.__limiter = Limiter(key_func=self.__get_api_key)
        self.MAX_FILE_SIZE_IN_MB = settings.MAX_FILE_SIZE_IN_MB
        self.__setting = settings
        self.setup_router()

    def __get_api_key(self, request: Request) -> str:
        api_key = request.headers.get("Api-Key")
        if not api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key is missing")
        return api_key

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
        @self.__limiter.limit(limit_value=self.__setting.RATE_LIMIT)
        async def create_chat_session(
            request: Request,
            tenant_id: Annotated[str | None, Header()],
            project_uuid: Annotated[str | None, Form()],
            message: Annotated[str | None, Form()],
            conversation_uuid: Annotated[str | None, Form()],
            is_stream: Annotated[bool | None, Form()] = False,
            files: List[UploadFile] | None = None,
            api_key: Annotated[str | None, Header()] = None,
            ):
            try:
                if api_key is None:
                    raise UnauthorizedAccess("API key is required")
                self.user_usecase.get_api_key(username=request.state.user, api_key=api_key)
                conversation = Conversation(project_id=project_uuid, conversation_uuid=conversation_uuid, message = message, tenant_id=tenant_id, is_stream=is_stream)
                if files is not None:
                    __check_file_validity(files)
                    for file in files:
                        document = Document(file=file)
                        document.set_multinancy_attr(project_uuid=project_uuid, tenant_id=tenant_id)
                        document_db = self.document_usecase.store_document(document=document)
                        langchain_document = await self.document_usecase.document_vectorization(document=document_db)
                        conversation.document_from_user.extend(langchain_document)
                
                conversation.project_id = project_uuid
                conversation.tenant_id = tenant_id
                if conversation.is_stream:
                    response_stream = self.conversation_usecase.stream_chat_agent(conversation=conversation)
                    return StreamingResponse(response_stream)  
                response, req_token, res_token = self.conversation_usecase.chat_with_agent(conversation=conversation)
                request.state.req_token = req_token
                request.state.res_token = res_token
                return Response(content=response)
            except UnauthorizedAccess as e:
                self.logger.error(str(e))
                raise HTTPException(status.HTTP_403_FORBIDDEN, "unauthorized access, please provide valid API key")
            except ResourceNotFound as e:
                self.logger.error(str(e))
                raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
            except FileTooLarge as e:
                self.logger.error(str(e))
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, str(e))
            except UnknownFileType as e:
                self.logger.error(str(e))
                raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
            except Exception as e:
                self.logger.error(str(e))
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "please try again later")
            
        def __check_file_validity(uploaded_files: List[UploadFile]):
            for file in uploaded_files:
                if file.size > self.MAX_FILE_SIZE_IN_MB * 1024 * 1025:
                    raise FileTooLarge("file is larger than {} MB".format(self.MAX_FILE_SIZE_IN_MB))
                content_type_buf = magic.from_buffer(file.file.read(4096), mime=True)
                content_type_ext = file.content_type
                self.logger.info("filename {} content-type from extension {}, content-type from binary properties {}".format(
                    file.filename, content_type_ext, content_type_buf,
                ))
                if content_type_buf != content_type_ext:
                    raise UnknownFileType("file type mismatch. extension: {}, binary's property: {}".format(content_type_ext, content_type_buf))

        @self.app.get("/v1/conversations")
        async def list_conversations(
            tenant_id: Annotated[str | None, Header()],
            project_uuid: Annotated[str | None, Query()] = None
        ) -> List[Conversation]:
            """
            List conversations filtered by tenant_id and optionally by project_uuid
            """
            try:
                if not tenant_id:
                    raise HTTPException(status.HTTP_400_BAD_REQUEST, "Tenant-Id header is required")
                
                # Create conversation filter object
                conversation_filter = Conversation(
                    tenant_id=tenant_id,
                    project_id=project_uuid
                )
                
                conversations = self.conversation_usecase.list_conversations(conversation_filter)
                
                if len(conversations) == 0:
                    self.logger.info("No conversations found for the given criteria")
                    raise HTTPException(status.HTTP_404_NOT_FOUND, "No conversations found")
                
                return conversations
                
            except HTTPException as e:
                raise e
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
            
        @self.app.get("/v1/download/{project_uuid}/{filename}")
        async def download_file(
            project_uuid: str,
            filename: str,
            tenant_id: Annotated[str, Header()]
        ):
            """
            Securely downloads a file for a given tenant and project.
            Prevents path traversal attacks.
            """
            try:
                if not tenant_id or not project_uuid:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tenant-Id and Project-UUID are required.")

                # Construct the expected directory for the tenant and project.
                # This helps in scoping the file access.
                project_dir = self.STORAGE_BASE_DIR / tenant_id / project_uuid
                
                # Construct the full path to the requested file.
                file_path = (project_dir / filename).resolve()

                # **Security Check**: Verify that the resolved file path is within the designated project directory.
                # This is the core of preventing directory traversal attacks.
                if not file_path.is_relative_to(project_dir.resolve()):
                    self.logger.warning(f"Path traversal attempt blocked for tenant '{tenant_id}', file '{filename}'")
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")
                
                if not file_path.exists() or not file_path.is_file():
                    self.logger.info(f"File not found: {file_path}")
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

                return FileResponse(path=file_path, media_type='application/octet-stream', filename=filename)

            except HTTPException as e:
                # Re-raise HTTPExceptions to let FastAPI handle them.
                raise e
            except Exception as e:
                self.logger.error(f"An error occurred during file download: {e}")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal error occurred.")
            
        @self.app.post("/v1/login")
        def user_login(request: LoginRequest):
            try:
                if request.username == "":
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="username is empty")
                if request.password == "":
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="password is empty")
                user_token = self.user_usecase.user_login(request=request)
                return user_token
            except UnauthorizedAccess as e:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str("username or password is incorrect"))
            except ResourceNotFound as e:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str("username or password is incorrect"))
            except HTTPException as e:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str("internal server error"))
            
        @self.app.post("/v1/user/access-token")
        def user_access_token(request: UserAccessTokenRequest, req: Request) -> UserAccessTokenResponse:
            try:
                if request.description == "":
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="username is empty")
                response = self.user_usecase.create_api_key(request, req.state.user)
                return response
            except Exception as e:
                self.logger.error(str(e))
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str("internal server error"))
            
        @self.app.post("/v1/internal/conversation")
        async def create_internal_chat_session(
            request: Request,
            api_key: Annotated[str,  Header()],
            project_id: Annotated[str,  Form()],
            conversation_uuid: Annotated[str, Form()],
            message: Annotated[str, Form()],
            username: Annotated[str | None, Header()] = None,
            files: List[UploadFile] | None = None,
        ):
            try:
                if username is None:
                    username = ""
                    self.user_usecase.get_api_key_internal(api_key=api_key)
                else:
                    self.user_usecase.get_api_key(username=username, api_key=api_key)
                conversation = Conversation(
                    project_id=project_id,
                    conversation_uuid=conversation_uuid,
                    message=message,
                    tenant_id=username,
                )
                if files is not None:
                    __check_file_validity(files)
                    for file in files:
                        document = Document(file=file)
                        document.set_multinancy_attr(project_uuid=project_id, tenant_id=username)
                        document_db = self.document_usecase.store_document(document=document)
                        langchain_document = await self.document_usecase.document_vectorization(document=document_db)
                        conversation.document_from_user.extend(langchain_document)

                conversation.project_id = project_id
                conversation.tenant_id = username
                response, req_token, res_token = self.conversation_usecase.chat_with_agent(conversation=conversation)
                request.state.req_token = req_token
                request.state.res_token = res_token
                return Response(content=response)
            except UnauthorizedAccess as e:
                self.logger.error(str(e))
                raise HTTPException(status.HTTP_403_FORBIDDEN, "unauthorized access, please provide valid API key")
            except ResourceNotFound as e:
                self.logger.error(str(e))
                raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
            except FileTooLarge as e:
                self.logger.error(str(e))
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, str(e))
            except UnknownFileType as e:
                self.logger.error(str(e))
                raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
            except Exception as e:
                self.logger.error(str(e))
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "please try again later")

        @self.app.post("/v2/internal/conversation")    
        async def create_internal_chat_session_v2(
            request: Request,
            api_key: Annotated[str,  Header()],
            project_id: Annotated[str,  Form()],
            conversation_uuid: Annotated[str, Form()],
            message: Annotated[str, Form()],
            username: Annotated[str | None, Header()] = None,
        ):
                if username is None:
                    username = ""
                    self.user_usecase.get_api_key_internal(api_key=api_key)
                else:
                    self.user_usecase.get_api_key(username=username, api_key=api_key)
                conversation = Conversation(
                    project_id=project_id,
                    conversation_uuid=conversation_uuid,
                    message=message,
                    tenant_id=username,
                )
                conversation.project_id = project_id
                conversation.tenant_id = username
                response = self.conversation_usecase.chat_with_agentv2(conversation=conversation)
                # request.state.req_token = req_token
                # request.state.res_token = res_token
                return response
            # except UnauthorizedAccess as e:
            #     self.logger.error(str(e))
            #     raise HTTPException(status.HTTP_403_FORBIDDEN, "unauthorized access, please provide valid API key")
            # except ResourceNotFound as e:
            #     self.logger.error(str(e))
            #     raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
            # except FileTooLarge as e:
            #     self.logger.error(str(e))
            #     raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, str(e))
            # except UnknownFileType as e:
            #     self.logger.error(str(e))
            #     raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
            # except Exception as e:
            #     self.logger.error(str(e))
            #     raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "please try again later")

        @self.app.get("/")
        def index():
            return "<p>welcome to document summarisation tools</p>"