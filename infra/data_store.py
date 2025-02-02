from infra.settings import Settings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from psycopg_pool import ConnectionPool

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
        self.checkpoint_saver_conn = PostgresSaver(conn=self.__connection_pool)
        self.checkpoint_saver_conn.setup()

    def get_connection(self):
        conn = self.__connection_pool.connection()
        return conn
            
    def get_vector_store(self):
        return self.__vector_store
    
    def get_checkpointer(self) -> BaseCheckpointSaver:
        return self.checkpoint_saver_conn
        

