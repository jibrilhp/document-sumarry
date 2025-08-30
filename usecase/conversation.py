import logging
from typing import List
import pandas as pd
from error.error import  DatabaseError, UnknownFileType
from entity.conversation import Conversation
from entity.document import Chat, UploadXAIFile, ShapSummary
from repository.chatbot import ChatBotRepository
from infra.generative_provider import GenerativeAdapter
from repository.document import DocumentRepository
from repository.chatbot_v2 import ChatBotV2Repository
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from repository.client_db import ClientDatabaseRepository
from io import BytesIO

class ConversationUsecase:
    def __init__(
        self, 
        chatbot_repository: ChatBotRepository, 
        document_repository: DocumentRepository, 
        chatbotv2_repository: ChatBotV2Repository,
        client_db_repository: ClientDatabaseRepository
    ):
        self.logger = logging.getLogger(__name__)
        self.chatbot_repository = chatbot_repository
        self.document_repository = document_repository
        self.__chatbotv2_repository = chatbotv2_repository
        self.__client_db_repository = client_db_repository

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

    def get_chat_history_v2(self, conversation: Conversation):
        config = {"configurable": {"thread_id": conversation.conversation_uuid}}
        chatbot = self.__chatbotv2_repository.get_chatbot_v2(conversation.conversation_uuid)
        if chatbot is None:
            self.logger.info("conversation for {} id is not found".format(conversation.conversation_uuid))
            chatbot = self.__chatbotv2_repository.create_chatbot_v2(conversation.conversation_uuid)
            self.logger.info("conversation for {} id is created".format(conversation.conversation_uuid))
        return self.__chatbotv2_repository.get_chat_history_v2(config, chatbot)
    
                                                                                                                                                                                                        
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
        client_db = self.__client_db_repository.get_client_db(conversation=conversation)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": conversation.conversation_uuid,
            },
            "recursion_limit": max(25, conversation.document_from_user.__len__() * 2)
        }
        response = chatbot.invoke(
            input={
                "messages": [HumanMessage(content=conversation.message)],
                "document_from_user": conversation.document_from_user if conversation.document_from_user is not None else [],
                "database_config": client_db,
            },
            config=config,
            stream_mode="values",
            output_keys=["agent_answer"]
        )
        return response.get("agent_answer")
    
    async def upload_xai_file(self, upload_xai_file: UploadXAIFile):
        try:
            # Reset file pointer to beginning
            await upload_xai_file.file.seek(0)
            
            # Read file content
            file_content = await upload_xai_file.file.read()
            
            # Check if file is empty
            if not file_content:
                raise ValueError("Uploaded file is empty")
            
            # Read file based on extension
            file_extension = upload_xai_file.file.filename.lower()
            
            if not file_extension.endswith('.csv'):
                raise UnknownFileType(message=f"Unsupported file type: {file_extension}. Only CSV files are supported.", file_type=file_extension)
            
            # Create BytesIO object for pandas to read
            file_buffer = BytesIO(file_content)
            df = pd.read_csv(file_buffer)
            
            # Validate DataFrame
            if df.empty:
                raise ValueError("CSV file contains no data")
            
            if len(df.columns) < 2:
                raise ValueError("CSV file must have at least 2 columns (features and predictions)")
            
            shap_means = df.drop(columns=["prediction"]).mean().sort_values(ascending=False)

            # Count prediction distribution
            pred_dist = df["prediction"].value_counts().to_dict()

            # Clean up feature names generically
            def clean_name(col):
                return (col.replace("shap_", "")
                        .replace("_", " ")
                        .title())

            # Top 3 positive
            top_pos = []
            for feature, val in shap_means.head(3).items():
                top_pos.append({"name": clean_name(feature), "mean_value": round(float(val), 3)})

            # Top 3 negative
            top_neg = []
            for feature, val in shap_means.tail(3).items():
                top_neg.append({"name": clean_name(feature), "mean_value": round(float(val), 3)})

            shap_summary = ShapSummary(
                file_name=upload_xai_file.file.filename,
                prediction_distribution=pred_dist,
                top_positive=top_pos,
                top_negative=top_neg
            )
            print(shap_summary.model_dump_json())
            return self.chatbot_repository.upload_xai_file(shap_summary)
            
        except UnknownFileType:
            raise
        except ValueError as e:
            self.logger.error(f"Validation error processing XAI file: {str(e)}")
            raise DatabaseError(f"Invalid XAI file: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error processing XAI file: {str(e)}")
            raise DatabaseError(f"Failed to process XAI file: {str(e)}")