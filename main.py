from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from infra.logging import setup_logging
from infra.settings import Settings
from infra.data_store import PostgresAdapter
from infra.storage import StorageRepository
from infra.generative_provider import GenerativeAdapter
from repository.document import DocumentRepository     
from repository.project import ProjectRepository
from repository.chatbot import ChatBotRepository
from repository.user import UserRepository
from repository.chatbot_v2 import ChatBotV2Repository
from repository.client_db import ClientDatabaseRepository
from usecase.document import DocumentUsecase
from usecase.project import ProjectUsecase
from usecase.conversation import ConversationUsecase
from usecase.user import UserUsecase
from handler.routes import Routes, AuthMiddleware
import logging

setup_logging()
logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "http://localhost:8083",
        "http://localhost:8088",
        "http://103.56.148.150:8088/",
        "http://103.56.148.150:8083/",
        "https://103.56.148.150:8088/"
        "https://license.dtskul-ai.ghanemtech.co.id/gpt"     # Alternative localhost
        # Add other allowed origins as needed
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
router = APIRouter()


logging.info("setup application...")
settings = Settings()
generative_adapter = GenerativeAdapter(settings)
postgres_adapter = PostgresAdapter(settings, generative_adapter.embedding_model)
documentRepository = DocumentRepository(db=postgres_adapter)
project_repository = ProjectRepository(db=postgres_adapter)
storage_repository = StorageRepository()
chatbot_repository = ChatBotRepository(postgres_adapter=postgres_adapter, generative_provider=generative_adapter)
user_repository = UserRepository(db=postgres_adapter)
chatbotv2_repository = ChatBotV2Repository(postgres_adapter=postgres_adapter, generative_provider=generative_adapter)
client_db_repository = ClientDatabaseRepository(db=postgres_adapter)
document_usecase = DocumentUsecase(document_repository=documentRepository, storage_repository=storage_repository, ollama_adapter=generative_adapter)
project_usecase = ProjectUsecase(project_repository=project_repository)
conversation_usecase = ConversationUsecase(chatbot_repository=chatbot_repository, document_repository=documentRepository, chatbotv2_repository=chatbotv2_repository, client_db_repository=client_db_repository)
user_usecase = UserUsecase(user_repository=user_repository, setting=settings)
routes = Routes(app=router, document_usecase=document_usecase, project_usecase=project_usecase, conversation_usecase=conversation_usecase, settings=settings, user_usecase=user_usecase)
app.add_middleware(AuthMiddleware, settings=settings)
app.include_router(router=router)