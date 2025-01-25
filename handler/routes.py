from flask import current_app, request, jsonify, Response
from usecase.document import DocumentUsecase
from usecase.project import ProjectUsecase
from entity.document import Document, DocumentDb, Chat
from entity.project import Project
from error.error import FileConflictDb, DatabaseError, ResourceNotFound


class Routes:
    def __init__(self, document_usecase: DocumentUsecase, project_usecase: ProjectUsecase):
        current_app.logger.setLevel("INFO")
        self.app = current_app
        self.document_usecase = document_usecase
        self.project_usecase = project_usecase

        @current_app.route("/v1/documents", methods=["POST"])
        def upload_document():
            try:
                tenant_id = request.headers["Tenant-Id"]
                file = request.files["document"]
                project_uuid = request.form["project"]
                self.app.logger.info("receive document {}".format(file.filename))
                document = Document(file=file)
                document.set_multinancy_attr(project_uuid=project_uuid, tenant_id=tenant_id)
                self.document_usecase.store_document(document=document)
                success = {"message": "document upload success"}
                return jsonify(success)
            except KeyError as e:
                self.app.logger.error(str(e))
                bad_request = {"error": "request is incomplete"}
                return jsonify(bad_request), 400
            except FileConflictDb as e:
                error = {"error": str(e)}
                return jsonify(error), 409

        @current_app.route("/v1/documents", methods=["GET"])
        def get_documents():
            tenant_id = request.headers.get("tenant_id")
            documentDb = DocumentDb()
            documentDb.set_multinancy_attr(project_uuid="", tenant_id=tenant_id)
            documents = self.document_usecase.get_document(documentDb)
            if documents.__len__() == 0:
                return jsonify({"error": "documents not found"}), 404
            listOfDocs = list()
            for document in documents:
                listOfDocs.append(document.__dict__)
            return jsonify(listOfDocs)

        @current_app.route("/v1/documents", methods=["DELETE"])
        def delete_document():
            try:
                tenant_id = request.headers.get("Tenant-Id")
                req = request.get_json()
                if not req:
                    return jsonify({"error": "empty request"}), 400
                document_uuid = req["uuid"]
                document = DocumentDb(uuid=document_uuid)
                document.set_multinancy_attr(project_uuid="", tenant_id=tenant_id)
                print(document.__dict__)
                deleted_document = self.document_usecase.delete_document(document=document)
                print(deleted_document.__dict__)
                if deleted_document.uuid == "":
                    return jsonify({"error": "document not found"}), 404
                return jsonify({"message":"document with {} uuid is deleted".format(document_uuid)})
            except ResourceNotFound as e:
                return jsonify({"error": str(e)}), 404
            except DatabaseError as e:
                self.app.logger.error(str(e))
                return jsonify({"error": "please try again later"}), 500
            
        @current_app.route("/v1/vector", methods=["POST"])
        async def vector_document():
            try:
                req = request.get_json()
                if not req:
                    return jsonify({"error": "empty request"}), 400
                document_uuid = req["uuid"]
                project_uuid = req["project_uuid"]
                tenant_id = request.headers["Tenant-Id"]
                document = DocumentDb(uuid=document_uuid)
                document.set_multinancy_attr(project_uuid=project_uuid, tenant_id=tenant_id)
                await self.document_usecase.document_vectorization(document=document)
                return jsonify({"message":"document with {} vectorization is completed".format(document_uuid)})
            except KeyError as e:
                return jsonify({"error": "{} not found".format(e)}), 400
            except ResourceNotFound as e:
                return jsonify({"error": str(e)}), 404
            except DatabaseError:
                return jsonify({"error": "please try again later"})
            
        @current_app.route("/v1/chat", methods=["POST"])
        def chat_generation():
            try:
                req = request.get_json()
                if not req:
                    return jsonify({"error": "empty request"}), 400
                user_chat: str = req["chat"]
                is_stream: bool = req["is_stream"]
                project_uuid: str = req["project_uuid"]
                tenant_id: str = request.headers["Tenant-Id"]
                chat = Chat(chat=user_chat, is_stream=is_stream)
                chat.set_multinancy_attr(project_uuid=project_uuid, tenant_id=tenant_id)
                if chat.is_stream:
                    return Response(
                        response=self.document_usecase.stream_chat_generation(chat=chat),
                        content_type="text/plain"
                    )
                result = self.document_usecase.chat_generation(chat=chat)
                return jsonify({"message": result})
            except Exception as e:
                self.app.logger.error(str(e))
                return jsonify({"error": "please try again later"}), 500
        
        @current_app.route("/v1/summarize", methods=["POST"])
        async def summarize_document():
            try:
                file = request.files["document"]
                self.app.logger.info("receive document {}".format(file.filename))
                document = Document(file=file)
                response = await self.document_usecase.summarize_document(document=document)
                return Response(
                    response=response,
                    content_type="text/plain"
                )
            except KeyError:
                bad_request = {"error": "filename not found"}
                return jsonify(bad_request), 400
        
        @current_app.route("/v1/projects", methods=["POST"])
        def create_project():
            try:
                req = request.get_json()
                project_name = req["name"]
                project = Project(name=project_name)
                project_uuid = self.project_usecase.create_project(project=project)
                return jsonify({"uuid":project_uuid})
            except KeyError:
                return jsonify({"error": "empty request"})

        @current_app.route("/v1/projects", methods=["GET"])
        def get_projects():
            try:
                projects = self.project_usecase.get_projects()
                if projects.__len__() == 0:
                    return jsonify({"error": "project not found"}), 404
                listOfProject = list()
                for project in projects:
                    listOfProject.append(project.__dict__)
                return jsonify(listOfProject)
            except ResourceNotFound as e:
                return jsonify({"error": str(e)}), 404
            except Exception as e:
                return jsonify({"error": "please try again later"}), 500
            
        @current_app.route("/v1/projects", methods=["DELETE"])
        def delete_project():
            try:
                req = request.get_json()
                uuid = req["uuid"]
                project = Project(uuid=uuid)
                self.project_usecase.delete_project(project=project)
                return jsonify({"message":"ok"})
            except KeyError:
                return jsonify({"error": "empty request"}), 400
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @current_app.route("/")
        def index():
            return "<p>welcome to document summarisation tools</p>"