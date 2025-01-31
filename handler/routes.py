from flask import current_app, request, jsonify
from usecase.document import DocumentUsecase
from usecase.project import ProjectUsecase
from usecase.conversation import ConversationUsecase
from entity.document import Document, DocumentDb, Chat
from entity.project import Project
from entity.conversation import Conversation
from error.error import FileConflictDb, DatabaseError, ResourceNotFound


class Routes:
    def __init__(self, document_usecase: DocumentUsecase, project_usecase: ProjectUsecase, conversation_usecase: ConversationUsecase):
        current_app.logger.setLevel("INFO")
        self.app = current_app
        self.document_usecase = document_usecase
        self.project_usecase = project_usecase
        self.conversation_usecase = conversation_usecase

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
                deleted_document = self.document_usecase.delete_document(document=document)
                if deleted_document.uuid == "":
                    return jsonify({"error": "document not found"}), 404
                return jsonify({"message":"document with {} uuid is deleted".format(document_uuid)})
            except ResourceNotFound as e:
                return jsonify({"error": str(e)}), 404
            except DatabaseError as e:
                self.app.logger.error(str(e))
                return jsonify({"error": "please try again later"}), 500
            
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
            
        @current_app.route("/v2/conversation", methods=["POST"])
        async def create_chat_session():
            try:
                tenant_id = request.headers["Tenant-Id"]
                project_uuid = request.form["project_uuid"]
                conversation_uuid = request.form["conversation_uuid"]
                message = request.form["message"]
                file = request.files.get("document")
                conversation = Conversation(
                    tenant_id=tenant_id, project_uuid=project_uuid, 
                    conversation_uuid=conversation_uuid, message=message
                )
                
                if file is not None:
                    document = Document(file=file)
                    document.set_multinancy_attr(project_uuid=project_uuid, tenant_id=tenant_id)
                    document_db = self.document_usecase.store_document(document=document)
                    langchain_document = await self.document_usecase.document_vectorization(document=document_db)
                    conversation.set_document_from_user(langchain_document)

                response_stream = self.conversation_usecase.chat_with_agent(conversation=conversation)
                return jsonify({"message": response_stream})
            except KeyError as e:
                self.app.logger.error(str(e))
                return jsonify({"error": str(e)}), 400

        @current_app.route("/")
        def index():
            return "<p>welcome to document summarisation tools</p>"