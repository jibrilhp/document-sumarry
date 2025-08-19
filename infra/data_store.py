from infra.settings import Settings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from psycopg_pool import ConnectionPool
from langchain_community.utilities import SQLDatabase
from typing import Set

class PostgresAdapter:
    def __init__(self, setting: Settings, embedding: Embeddings):
        self.__dsn = "postgresql://{}:{}@{}:{}/{}".format(
            setting.DB_USER,
            setting.DB_PASSWORD,
            setting.DB_HOST,
            setting.DB_PORT,
            setting.DB_NAME,
        )
        self._connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }
        self.__connection_pool = ConnectionPool(conninfo=self.__dsn, max_size=20, kwargs=self._connection_kwargs)
        self.__vector_store = PGVector(
            embeddings=embedding,
            connection=self.__dsn,
            use_jsonb=True,
            async_mode=False,
            collection_name=setting.COLLECTION_NAME
        )
        self.__checkpoint_saver_conn = PostgresSaver(conn=self.__connection_pool)
        self.__checkpoint_saver_conn.setup()

    def get_connection(self):
        conn = self.__connection_pool.connection()
        return conn
            
    def get_vector_store(self):
        return self.__vector_store
    
    def get_checkpointer(self) -> BaseCheckpointSaver:
        return self.__checkpoint_saver_conn

    def create_sql_database(self, db_uri: str, include_tables: Set[str]) -> SQLDatabase:
        sql_db =  SQLDatabase.from_uri(db_uri)
        sql_db._include_tables = include_tables
        return sql_db
