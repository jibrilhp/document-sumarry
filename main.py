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
from usecase.document import DocumentUsecase
from usecase.project import ProjectUsecase
from usecase.conversation import ConversationUsecase
from handler.routes import Routes
import logging

setup_logging()
logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Alternative localhost
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
document_usecase = DocumentUsecase(document_repository=documentRepository, storage_repository=storage_repository, ollama_adapter=generative_adapter)
project_usecase = ProjectUsecase(project_repository=project_repository)
conversation_usecase = ConversationUsecase(ollama_adapter=generative_adapter, chatbot_repository=chatbot_repository, document_repository=documentRepository)
routes = Routes(app=router, document_usecase=document_usecase, project_usecase=project_usecase, conversation_usecase=conversation_usecase)
app.include_router(router=router)