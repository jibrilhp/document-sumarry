from repository.data_store import PostgresAdapter
from entity.project import Project
from flask import current_app
from uuid import uuid5, NAMESPACE_X500
from psycopg2.errors import UniqueViolation
from error.error import FileConflictDb, DatabaseError, ResourceNotFound
from typing import List

class ProjectRepository:
    def __init__(self, db: PostgresAdapter):
        self.app = current_app
        self.cursor = db.get_cursor()
        self.connection = db.get_connection()

    def create_project(self, project: Project) -> str:
        uuid = uuid5(namespace=NAMESPACE_X500, name=project.name)
        sql = "insert into projects(uuid, name) values (%s, %s)"
        data = (uuid.__str__(), project.name)
        try:
            self.cursor.execute(query=sql, vars=data)
            self.connection.commit()
            self.app.logger.info("project {} created with uuid {}".format(uuid.__str__(), project.name))
            return uuid
        except UniqueViolation:
            self.connection.rollback()
            raise FileConflictDb("{} already exist in database".format(project.name))
        except Exception:
            self.connection.rollback()
            raise DatabaseError("please try again later")
        
    def get_project(self, project: Project) -> Project:
        sql = "select uuid, name from projects where uuid = ?"
        data = (project.uuid,)
        try:
            self.cursor.execute(query=sql, vars=data)
            results = self.cursor.fetchall()
            if results.__len__() == 0:
                e = ResourceNotFound("project not found")
                self.app.logger.error(str(e))
                raise e
            for row in results:
                uuid, name = row
                project = Project(uuid=uuid, name=name)
                return project
        except Exception:
            raise DatabaseError("please try again later")
        

    def get_projects(self) -> List[Project]:
        sql = "select uuid, name from projects"
        projects: List[Project] = list()
        try:
            self.cursor.execute(query=sql)
            results = self.cursor.fetchall()
            if results.__len__() == 0:
                e = ResourceNotFound("project not found")
                self.app.logger.error(str(e))
                raise ResourceNotFound("project not found")
            for row in results:
                uuid, name = row
                project = Project(uuid=uuid, name=name)
                projects.append(project)
                return projects
        except Exception as e:
            self.app.logger.error(e)
            raise DatabaseError("please try again later")
        
    def delete_document(self, project: Project):
        sql = "DELETE FROM projects WHERE uuid = %s"
        data = (project.uuid,)
        self.app.logger.info("deleting project with uuid: {}".format(project.uuid))
        try:
            self.cursor.execute(query=sql, vars=data)
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            self.app.logger.error(e)
            raise DatabaseError("please try again later")
