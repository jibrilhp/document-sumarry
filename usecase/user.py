from entity.user import LoginRequest, UserToken, UserAccessTokenRequest, UserAccessTokenResponse
from error.error import UnauthorizedAccess
from repository.user import UserRepository
from infra.settings import Settings
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
import jwt
import logging
import hashlib

class UserUsecase:
    def __init__(self, user_repository:UserRepository, setting: Settings):
        self.__pwd_ctx = CryptContext(schemes="bcrypt")
        self.__user_repository = user_repository
        self.__setting = setting
        self.__logging = logging.getLogger(__file__)

    def user_login(self, request: LoginRequest) -> UserToken:
        try:
            user = self.__user_repository.get_user(request=request)
            is_password_verified = self.__pwd_ctx.verify(request.password, user.password)
            if not is_password_verified:
                raise UnauthorizedAccess("unauthorized access")
            access_token_payload = {
                "sub": user.username,
                "iss": "document-summary-app",
                "iat": datetime.now(timezone.utc)
            }
            access_token = self.__create_access_token(access_token_payload)
            return UserToken(username=user.username, token=access_token)
        except UnauthorizedAccess as e:
            self.__logging.error(str(e))
            raise


    def __create_access_token(self, access_token_payload: dict, expires_delta: timedelta | None = None) -> str:
        to_encode = access_token_payload.copy()
        if expires_delta is None:
            expire = datetime.now(timezone.utc) + timedelta(days=1)
        else:
            expire = datetime.now(timezone.utc) + expires_delta
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(payload=to_encode, key=self.__setting.SECRET_KEY, algorithm=self.__setting.JWT_ALGORITHM)
        return encoded_jwt

    def create_api_key(self, request: UserAccessTokenRequest, username: str) -> UserAccessTokenResponse:
        api_key = self.__pwd_ctx.hash(f"{datetime.now(timezone.utc).now().microsecond}-{self.__setting.SECRET_KEY}")
        try:
            self.__user_repository.store_api_key(request=request, token=api_key, username=username)
            response = UserAccessTokenResponse(api_token=api_key, description=request.description, message="API key created successfully. Store it securely as you can't recover it again")
            return response
        except Exception as e:
            self.__logging.error(str(e))
            raise

    def get_api_key(self, username: str, api_key: str) -> str:
        api_key = self.__user_repository.get_api_key(username=username, api_key=api_key)
        return api_key
        
        

        
