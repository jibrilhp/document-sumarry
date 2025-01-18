from flask import current_app
import psycopg2
import uuid
from entity.document import Document

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