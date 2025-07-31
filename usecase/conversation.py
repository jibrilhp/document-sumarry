import logging
from typing import List
from error.error import  DatabaseError
from entity.conversation import Conversation
from entity.document import Chat
from repository.chatbot import ChatBotRepository
from infra.generative_provider import GenerativeAdapter
from repository.document import DocumentRepository
from repository.chatbot_v2 import ChatBotV2Repository
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage

class ConversationUsecase:
    def __init__(self, ollama_adapter: GenerativeAdapter, chatbot_repository: ChatBotRepository, document_repository: DocumentRepository, chatbotv2_repository: ChatBotV2Repository):
        self.logger = logging.getLogger(__name__)
        self.ollama_adapter = ollama_adapter
        self.chatbot_repository = chatbot_repository
        self.document_repository = document_repository
        self.__chatbotv2_repository = chatbotv2_repository

    def __retrieve_chatbot(self, conversation: Conversation):
        self.logger.info("conversation request with conversation_uuid {}".format(conversation.conversation_uuid))
        chatbot = self.chatbot_repository.get_chatbot(conversation.conversation_uuid)
        if chatbot is None:
            self.logger.info("conversation for {} id is not found".format(conversation.conversation_uuid))
            chatbot = self.chatbot_repository.create_chatbot(conversation.conversation_uuid)
            self.logger.info("conversation for {} id is created".format(conversation.conversation_uuid))
        return chatbot

    def chat_with_agent(self, conversation: Conversation):
        if conversation.document_from_user is None:
            conversation.document_from_user = list()
        chatbot = self.__retrieve_chatbot(conversation=conversation)
        config: RunnableConfig = {
            "recursion_limit": max(25, conversation.document_from_user.__len__() * 2),
            "configurable": {
                "thread_id": conversation.conversation_uuid
            }
        }
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
        return response.get("answer"), response.get("request_token_count"), response.get("response_token_count")

    
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
    
                                                                                                                                                                                                        
    def list_conversations(self, conversation_filter: Conversation) -> List[Conversation]:
        """
        List conversations based on tenant_id and optionally project_id
        """
        try:
            # This would interact with your database layer
            # The exact implementation depends on your data access pattern
            conversations = self.conversation_repository.get_conversations_by_filter(
                tenant_id=conversation_filter.tenant_id,
                project_id=conversation_filter.project_id
            )
            return conversations
        except Exception as e:
            self.logger.error(f"Error listing conversations: {str(e)}")
            raise DatabaseError("Failed to retrieve conversations")
        
    def __retrieve_chatbotv2(self, conversation: Conversation):
        self.logger.info("conversation request using v2 conversation uuid {}".format(conversation.conversation_uuid))
        chatbot = self.__chatbotv2_repository.get_chatbot_v2(thread_id=conversation.conversation_uuid)
        if chatbot is None:
            self.logger.info("conversation v2 for {} id is not found".format(conversation.conversation_uuid))
            chatbot = self.__chatbotv2_repository.create_chatbot_v2(conversation.conversation_uuid)
            self.logger.info("conversation for {} id is created".format(conversation.conversation_uuid))
        return chatbot
    
    def chat_with_agentv2(self, conversation: Conversation):
        chatbot = self.__retrieve_chatbotv2(conversation=conversation)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": conversation.conversation_uuid
            }
        } 
        response = chatbot.invoke(
            input={"messages": [HumanMessage(content=conversation.message)]},
            config=config,
            stream_mode="values",
            output_keys=["agent_answer"]
        )
        return response.get("agent_answer")
    