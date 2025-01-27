from flask import Flask
import json
from repository.data_store import PostgresAdapter, InMemoryVector, PostgresCheckpointer
from repository.document import DocumentRepository
from repository.storage import StorageRepository
from repository.ollama import OllamaAdapter
from repository.data_store import PGVectorAdapter
from repository.project import ProjectRepository
from repository.chat_bot import ChatBotRepository
from usecase.document import DocumentUsecase
from usecase.project import ProjectUsecase
from usecase.conversation import Conversation
from handler.routes import Routes

app = Flask(__name__)

if __name__ == "__main__":
    app.config.from_file("env.json", load=json.load)
    with app.app_context():
        pg_checkpointer = PostgresCheckpointer()
        ollama_adapter = OllamaAdapter()
        pg_vector_adapter = PGVectorAdapter(ollama_adapter.embedding_model)
        inmemory_vector_adapter = InMemoryVector(ollama_adapter.embedding_model)
        postgres_adapter = PostgresAdapter()
        documentRepository = DocumentRepository(db=postgres_adapter, pgvector=pg_vector_adapter, inmemory_vector=inmemory_vector_adapter)
        project_repository = ProjectRepository(db=postgres_adapter)
        storage_repository = StorageRepository()
        chatbot_repository = ChatBotRepository(postgres_checkpointer=pg_checkpointer)
        document_usecase = DocumentUsecase(document_repository=documentRepository, storage_repository=storage_repository, ollama_adapter=ollama_adapter)
        project_usecase = ProjectUsecase(project_repository=project_repository)
        conversation_usecase = Conversation(ollama_adapter=ollama_adapter, chatbot_repository=chatbot_repository, document_repository=documentRepository)
        routes = Routes(document_usecase=document_usecase, project_usecase=project_usecase, conversation_usecase=conversation_usecase)
    app.run(debug=True)
