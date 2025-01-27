from flask import current_app
from entity.conversation import ConversationIdentity
from entity.document import Chat
from repository.chat_bot import ChatBotRepository
from repository.ollama import OllamaAdapter
from repository.document import DocumentRepository

class Conversation:
    def __init__(self, ollama_adapter: OllamaAdapter, chatbot_repository: ChatBotRepository, document_repository: DocumentRepository):
        self.app = current_app
        self.ollama_adapter = ollama_adapter
        self.chatbot_repository = chatbot_repository
        self.document_repository = document_repository

    def chat_with_agent(self, conversion_identity: ConversationIdentity):
        self.app.logger.info("conversation request with hash: {}".format(conversion_identity.conversation_uuid))
        conversation_chatbot = self.chatbot_repository.get_chat_state(thread_id=conversion_identity.conversation_uuid)
        if conversation_chatbot is None:
            self.app.logger.info("conversation for {} id is not found".format(conversion_identity.conversation_uuid))
            conversation_chatbot = self.chatbot_repository.create_chat_state(thread_id=conversion_identity.conversation_uuid, runnable=self.ollama_adapter.chat_bot)
            self.app.logger.info("conversation for {} id is created".format(conversion_identity.conversation_uuid))
        config = {"configurable": {"thread_id": conversion_identity.conversation_uuid}}
        chat = Chat(chat=conversion_identity.message, is_stream=False)
        chat.set_multinancy_attr(project_uuid=conversion_identity.project_uuid, tenant_id=conversion_identity.tenant_id)
        relevant_document = self.document_repository.find_relevant_document(chat=chat)
        events = conversation_chatbot.invoke(
            {"messages": [{"role": "user", "content": self.ollama_adapter.generate_prompt(chat=conversion_identity, similiar_documents=relevant_document)}]}, 
            config=config,
            stream_mode="values"
            )
        response_content = events["messages"][-1].content
        return response_content