from infra.data_store import PostgresAdapter
from entity.user import LoginRequest, User, UserAccessTokenRequest
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
            
    def store_api_key(self, request: UserAccessTokenRequest, token: str, username: str) -> None:
        with self.__db_adapter.get_connection() as conn:
            sql = "INSERT INTO api_keys (api_key, description, user_id) VALUES (%s, %s, %s)"
            data = (token, request.description, username)
            try:
                conn.execute(sql, data)
            except Exception as e:
                self.logger.error(str(e))
                raise DatabaseError("database has error when storing API key")
            
    def get_api_key(self, username: str, api_key: str) -> str:
        with self.__db_adapter.get_connection() as conn:
            sql = "SELECT api_key FROM api_keys WHERE user_id = %s AND api_key = %s LIMIT 1"
            data = (username,api_key)
            try:
                result = conn.execute(sql, data).fetchone()
                if result is None:
                    raise ResourceNotFound("API key not found for user")
                api_key: str = result
                return api_key
            except ResourceNotFound as e:
                self.logger.error(str(e))
                raise 
            except Exception as e:
                self.logger.error(str(e))
                raise DatabaseError("database has error when querying API key")
