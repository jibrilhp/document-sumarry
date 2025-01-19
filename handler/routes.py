from flask import current_app, request, jsonify, Response
from usecase.document import DocumentUsecase
from entity.document import Document, DocumentDb, Chat
from error.error import FileConflictDb, DatabaseError, ResourceNotFound


class Routes:
    def __init__(self, document_usecase: DocumentUsecase):
        current_app.logger.setLevel("INFO")
        self.app = current_app
        self.document_usecase = document_usecase

        @current_app.route("/v1/documents", methods=["POST"])
        def upload_document():
            try:
                file = request.files["document"]
                self.app.logger.info("receive document {}".format(file.filename))
                document = Document(file=file)
                self.document_usecase.store_document(document=document)
                success = {"message": "document upload success"}
                return jsonify(success)
            except KeyError:
                bad_request = {"error": "filename not found"}
                return jsonify(bad_request), 400
            except FileConflictDb as e:
                error = {"error": str(e)}
                return jsonify(error), 409

        @current_app.route("/v1/documents", methods=["GET"])
        def get_documents():
            documents = self.document_usecase.get_document()
            if documents.__len__() == 0:
                return jsonify({"error": "documents not found"}), 404
            listOfDocs = list()
            for document in documents:
                listOfDocs.append(document.__dict__)
            return jsonify(listOfDocs)

        @current_app.route("/v1/documents", methods=["DELETE"])
        def delete_document():
            try:
                req = request.get_json()
                if not req:
                    return jsonify({"error": "empty request"}), 400
                document_uuid = req["uuid"]
                document = DocumentDb(uuid=document_uuid)
                self.document_usecase.delete_document(document=document)
                return jsonify({"message":"document with {} uuid is deleted".format(document_uuid)})
            except ResourceNotFound as e:
                return jsonify({"error": str(e)}), 404
            except DatabaseError:
                return jsonify({"error": "please try again later"})
            
        @current_app.route("/v1/vector", methods=["POST"])
        async def vector_document():
            try:
                req = request.get_json()
                if not req:
                    return jsonify({"error": "empty request"}), 400
                document_uuid = req["uuid"]
                document = DocumentDb(uuid=document_uuid)
                await self.document_usecase.document_vectorization(document=document)
                return jsonify({"message":"document with {} vectorization is completed".format(document_uuid)})
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
                chat = Chat(chat=user_chat, is_stream=is_stream)
                if chat.is_stream:
                    return Response(
                        response=self.document_usecase.stream_chat_generation(chat=chat),
                        content_type="text/plain"
                    )
                result = self.document_usecase.chat_generation(chat=chat)
                return jsonify({"message": result})
            except Exception:
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
            
        @current_app.route("/")
        def index():
            return "<p>welcome to document summarisation tools</p>"