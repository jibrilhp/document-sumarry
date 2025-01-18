from flask import Flask
import json
from repository.postgres import PostgresAdapter
from repository.document import DocumentRepository
from repository.storage import StorageRepository
from usecase.document import DocumentUsecase
from handler.routes import Routes

app = Flask(__name__)

if __name__ == "__main__":
    app.config.from_file("env.json", load=json.load)
    with app.app_context():
        postgres_adapter = PostgresAdapter()
        documentRepository = DocumentRepository(db=postgres_adapter)
        storage_repository = StorageRepository()
        document_usecase = DocumentUsecase(document_repository=documentRepository, storage_repository=storage_repository)
        routes = Routes(document_usecase=document_usecase)
    app.run(debug=True)
