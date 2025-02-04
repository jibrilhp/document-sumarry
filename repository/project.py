from infra.data_store import PostgresAdapter
from entity.project import Project
from uuid import uuid5, NAMESPACE_X500
# from psycopg2.errors import UniqueViolation
from psycopg.errors import UniqueViolation
from error.error import FileConflictDb, DatabaseError, ResourceNotFound
from typing import List
import logging

class ProjectRepository:
    def __init__(self, db: PostgresAdapter):
        self.logger = logging.getLogger(__name__)
        self.db_adapter = db

    def create_project(self, project: Project) -> Project:
        with self.db_adapter.get_connection() as conn:
            uuid = uuid5(namespace=NAMESPACE_X500, name=project.name)
            sql = "insert into projects(uuid, name) values (%s, %s)"
            data = (uuid.__str__(), project.name)
            try:
                conn.execute(sql, data)
                conn.commit()
                self.logger.info("project {} created with uuid {}".format(uuid.__str__(), project.name))
                project.uuid = uuid.__str__()
                return project
            except UniqueViolation:
                conn.rollback()
                raise FileConflictDb("{} already exist in database".format(project.name))
            except Exception as e:
                self.logger.error(str(e))
                conn.rollback()
                raise DatabaseError("please try again later")
        
    def get_project(self, project: Project) -> Project:
        with self.db_adapter.get_connection() as conn:
            sql = "select uuid, name from projects where uuid = %s"
            data = (project.uuid,)
            try:
                results = conn.execute(sql, data).fetchall()
                for row in results:
                    uuid, name = row
                    project = Project(uuid=uuid, name=name)
                    return project
            except Exception as e:
                self.logger.error(str(e))
                raise DatabaseError("please try again later")
        

    def get_projects(self) -> List[Project]:
        with self.db_adapter.get_connection() as conn:
            sql = "select uuid, name from projects"
            projects: List[Project] = list()
            try:
                results = conn.execute(sql).fetchall()
                for row in results:
                    uuid, name = row
                    project = Project(uuid=uuid, name=name)
                    projects.append(project)
                return projects
            except Exception as e:
                self.logger.error(e)
                conn.rollback()
                raise DatabaseError("please try again later")
        
    def delete_document(self, project: Project):
        with self.db_adapter.get_connection() as conn:
            sql = "DELETE FROM projects WHERE uuid = %s"
            data = (project.uuid,)
            self.logger.info("deleting project with uuid: {}".format(project.uuid))
            try:
                conn.execute(sql, data)
                conn.commit()
            except Exception as e:
                conn.rollback()
                self.logger.error(e)
                raise DatabaseError("please try again later")
