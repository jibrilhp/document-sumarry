from flask import current_app
from entity.conversation import Conversation
from entity.document import Chat
from repository.chatbot import ChatBotRepository
from infra.generative_provider import GenerativeAdapter
from repository.document import DocumentRepository

class ConversationUsecase:
    def __init__(self, ollama_adapter: GenerativeAdapter, chatbot_repository: ChatBotRepository, document_repository: DocumentRepository):
        self.app = current_app
        self.ollama_adapter = ollama_adapter
        self.chatbot_repository = chatbot_repository
        self.document_repository = document_repository

    def chat_with_agent(self, conversation: Conversation):
        self.app.logger.info("conversation request with conversation_uuid {}".format(conversation.conversation_uuid))
        chatbot = self.chatbot_repository.get_chatbot(conversation.conversation_uuid)
        if chatbot is None:
            self.app.logger.info("conversation for {} id is not found".format(conversation.conversation_uuid))
            chatbot = self.chatbot_repository.create_chatbot(conversation.conversation_uuid)
            self.app.logger.info("conversation for {} id is created".format(conversation.conversation_uuid))
        config = {"configurable": {"thread_id": conversation.conversation_uuid}}
        response = chatbot.invoke(
            input={"question": conversation.message, "document_from_user": conversation.document_from_user},
            config=config,
            stream_mode="values"
        )
        return response.get("answer")
        