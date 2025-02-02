from repository.project import ProjectRepository
from entity.project import Project
from typing import List

class ProjectUsecase:
    def __init__(self, project_repository: ProjectRepository):
        self.project_repository = project_repository

    def create_project(self, project: Project) -> Project:
        return self.project_repository.create_project(project=project)
    
    def get_projects(self) -> List[Project]:
        return self.project_repository.get_projects()
    
    def delete_project(self, project: Project):
        project_from_db = self.project_repository.get_project(project=project)
        self.project_repository.delete_document(project=project_from_db)
        return