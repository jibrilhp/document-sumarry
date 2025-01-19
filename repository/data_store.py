from flask import current_app
import psycopg2
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_core.embeddings import Embeddings
from typing import List
from langchain_community.vectorstores import InMemoryVectorStore

class PostgresAdapter:
    def __init__(self):
        self.connection = psycopg2.connect(
            host="{}".format(current_app.config["DB_HOST"], 
            port = current_app.config["DB_PORT"]),
            database=current_app.config["DB_NAME"],
            user=current_app.config["DB_USER"],
            password=current_app.config["DB_PASSWORD"]
        )
        self.cursor = self.connection.cursor()
        current_app.logger.info("database connected")

    def get_cursor(self):
        return self.cursor
    
    def get_connection(self):
        return self.connection

    def close(self):
        self.cursor.close()
        self.connection.close()

class PGVectorAdapter:
    def __init__(self, embedding: Embeddings):
        self._connection = "postgresql+psycopg://{}:{}@{}:{}/{}".format(
            current_app.config["DB_USER"],
            current_app.config["DB_PASSWORD"],
            current_app.config["DB_HOST"],
            current_app.config["DB_PORT"],
            current_app.config["DB_NAME"]
        )
        current_app.logger.info(self._connection)
        self.vector_store = PGVector(
            embeddings=embedding,
            collection_name=current_app.config["COLLECTION_NAME"],
            connection=self._connection,
            use_jsonb=True
        )
        current_app.logger.info("pg vector ready")


class InMemoryVector:
    def __init__(self, embedding: Embeddings):
        self.in_memory_vector_store = InMemoryVectorStore(embedding=embedding)