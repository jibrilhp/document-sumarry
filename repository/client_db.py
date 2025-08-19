from infra.data_store import PostgresAdapter
from entity.conversation import Conversation
from typing import List
from entity.conversation import DatabaseConfig

class ClientDatabaseRepository:
    def __init__(self, db: PostgresAdapter):
        self.__db_adapter = db

    def get_client_db(self, conversation: Conversation) -> List[DatabaseConfig]:
        with self.__db_adapter.get_connection() as conn:
            sql = "select d.dataset_name, db_type, db_host, db_port, db_name, d.table_name, db_username, db_password from databases inner join datasets d on d.database_id = databases.id where d.created_by = %s ;"
            data = (conversation.tenant_id,)
            results = conn.cursor().execute(sql, data).fetchall()
            results_db: List[DatabaseConfig] = list()
            for result in results:
                db_config = DatabaseConfig(
                    dataset_name=result[0],
                    db_type=result[1],
                    db_host=result[2],
                    db_port=result[3],
                    db_name=result[4],
                    table_name=result[5],
                    db_username=result[6],
                    db_password=result[7]
                )
                db_config.db_uri = db_config.set_db_uri()
                results_db.append(db_config)
            return results_db