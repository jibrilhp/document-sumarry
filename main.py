from fastapi import FastAPI, APIRouter
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_client import Counter
from infra.logging import setup_logging
from infra.settings import Settings
from infra.data_store import PostgresAdapter
from infra.storage import StorageRepository
from infra.generative_provider import GenerativeAdapter
from infra.instrumentation import request_token_counter, response_token_counter
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
router = APIRouter()
instrumentator = Instrumentator()
instrumentator.add(
    request_token_counter()  # Your custom metric functions
).add(
    response_token_counter()
)
instrumentator.instrument(app).expose(app)  # Instrument before defining routes

# NOW you can add custom metrics


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
routes = Routes(app=router, document_usecase=document_usecase, project_usecase=project_usecase, conversation_usecase=conversation_usecase, settings=settings)
app.include_router(router=router)