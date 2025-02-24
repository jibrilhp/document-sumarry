import logging
from entity.conversation import Conversation
from entity.document import Chat
from repository.chatbot import ChatBotRepository
from infra.generative_provider import GenerativeAdapter
from repository.document import DocumentRepository

class ConversationUsecase:
    def __init__(self, ollama_adapter: GenerativeAdapter, chatbot_repository: ChatBotRepository, document_repository: DocumentRepository):
        self.logger = logging.getLogger(__name__)
        self.ollama_adapter = ollama_adapter
        self.chatbot_repository = chatbot_repository
        self.document_repository = document_repository

    def __retrieve_chatbot(self, conversation: Conversation):
        self.logger.info("conversation request with conversation_uuid {}".format(conversation.conversation_uuid))
        chatbot = self.chatbot_repository.get_chatbot(conversation.conversation_uuid)
        if chatbot is None:
            self.logger.info("conversation for {} id is not found".format(conversation.conversation_uuid))
            chatbot = self.chatbot_repository.create_chatbot(conversation.conversation_uuid)
            self.logger.info("conversation for {} id is created".format(conversation.conversation_uuid))
        return chatbot

    def chat_with_agent(self, conversation: Conversation):
        chatbot = self.__retrieve_chatbot(conversation=conversation)
        config = {"configurable": {"thread_id": conversation.conversation_uuid}}
        response = chatbot.invoke(
            input={"conversation": [{
                "role": "user", "content": conversation.message,
                }], 
                "document_from_user": conversation.document_from_user,
                "tenant_id": conversation.tenant_id,
                "project_uuid": conversation.project_id,
                },
            config=config,
            stream_mode="values",
            output_keys=["answer", "request_token_count", "response_token_count"]
        )
        print(response)
        return response.get("answer")

    
    def stream_chat_agent(self, conversation: Conversation):
        chatbot = self.__retrieve_chatbot(conversation=conversation)
        config = {"configurable": {"thread_id": conversation.conversation_uuid}}
        if conversation.is_stream:
            response = chatbot.stream(
                input={"conversation": [{
                    "role": "user", 
                    "content": conversation.message,
                    }], 
                    "document_from_user": conversation.document_from_user,
                    "tenant_id": conversation.tenant_id,
                    "project_uuid": conversation.project_id,
                    },
                config=config,
                stream_mode="values",
                output_keys=["answer", "request_token_count", "response_token_count"]
            )
            for r in response:
                print(r)
                yield r.get("answer")

    def get_chat_history(self, conversation: Conversation):
        config = {"configurable": {"thread_id": conversation.conversation_uuid}}
        chatbot = self.chatbot_repository.get_chatbot(conversation.conversation_uuid)
        if chatbot is None:
            self.logger.info("conversation for {} id is not found".format(conversation.conversation_uuid))
            chatbot = self.chatbot_repository.create_chatbot(conversation.conversation_uuid)
            self.logger.info("conversation for {} id is created".format(conversation.conversation_uuid))
        return self.chatbot_repository.get_chat_history(config, chatbot)
        