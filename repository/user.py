from infra.data_store import PostgresAdapter
from entity.user import LoginRequest, User
from error.error import ResourceNotFound, DatabaseError
import logging

class UserRepository:
    def __init__(self, db: PostgresAdapter):
        self.logger = logging.getLogger(__file__)
        self.__db_adapter = db

    def get_user(self, request: LoginRequest) -> User:
        with self.__db_adapter.get_connection() as conn:
            sql = "SELECT username, email, password FROM users WHERE username = %s LIMIT 1"
            data = (request.username,)
            try:
                result = conn.execute(sql, data).fetchone()
                if result is None:
                    raise ResourceNotFound("user not found")
                username, email, password = result
                user = User(username=username, email=email, password=password)
                return user
            except ResourceNotFound as e:
                self.logger.error(str(e))
                raise 
            except Exception as e:
                self.logger.error(str(e))
                raise DatabaseError("database has error when querying user")
