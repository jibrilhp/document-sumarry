from typing import Dict, Literal, List
from entity.conversation import ConversationalChatbot, ConversationStateV2
from entity.tools import VectorSearchInput, VectorSearchOutput
from infra.generative_provider import GenerativeAdapter
from infra.data_store import PostgresAdapter
from infra.prompt_loader import PromptLoader
import logging
from langgraph.graph import START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.prompts import ChatPromptTemplate
from entity.conversation import StateV2, AgentResponseV2, RouterOutputV2, DatabaseConfig, ChartData
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import RunnableSerializable
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import Runnable
from langchain_core.tools.base import BaseTool
from langchain_tavily import TavilySearch, TavilyExtract
from langchain_core.tools import tool
from langmem.short_term import SummarizationNode
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.messages import AIMessage
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from pandas import read_json
from io import StringIO

class ChatBotV2Repository:
    def __init__(
        self,
        generative_provider: GenerativeAdapter,
        postgres_adapter: PostgresAdapter,
        prompts_file_path: str = "config-rag.yaml"
    ):
        self.__generative_adapter = generative_provider
        self.__pgvector = postgres_adapter.get_vector_store()
        self.__checkpointer = postgres_adapter.get_checkpointer()
        self.__logger = logging.getLogger(__name__)
        self.__chat_states: Dict[str, CompiledStateGraph] = {}
        self.__create_sql_database = postgres_adapter.create_sql_database
        self.__prompt_loader = PromptLoader(prompts_file_path)

    def create_chatbot_v2(self, thread_id: str):
        chatbot = ConversationalChatbot()
        chatbot.add_edges(START, "memory_retrieval")
        chatbot.add_edges("memory_retrieval", "summarize_conversation")
        chatbot.add_conditional_edges("summarize_conversation", self.__should_summarize)

        chatbot.add_node("memory_retrieval", self.__memory_retrieval_node)
        chatbot.add_node("llm_router", self.__create_workflow_routing_chain)
        chatbot.add_node("specific_assistance", self.__specific_knowledge_agent_chain)
        chatbot.add_node("sql_assistant", self.__sql_agent_chain)
        chatbot.add_node("chart_generator", self.__generate_chart)
        chatbot.add_node("pandas_dataframe_agent", self.__pandas_dataframe_chain)
       
        chatbot.add_node("initial_summary", self.__generate_initial_summary)
        chatbot.add_node("summary_refinement", self.__generate_summary_refinement)

        chatbot.add_node("summarize_conversation", self.__summarization_node())
        chatbot.add_node("memory_update", self.__memory_update_node)
        chatbot.add_node("store_response", self.__store_response)

        chatbot.add_conditional_edges("llm_router", self.__llm_router)
        chatbot.add_conditional_edges("initial_summary", self.__should_refine)
        chatbot.add_conditional_edges("summary_refinement", self.__should_refine)

        chatbot.add_edges("specific_assistance", "memory_update")
        chatbot.add_edges("sql_assistant", "chart_generator")
        chatbot.add_edges("chart_generator", "memory_update")
        chatbot.add_edges("pandas_dataframe_agent", "memory_update")
        chatbot.add_edges("memory_update", "store_response")
        chatbot.add_edges("store_response", END)
        compiled_graph = chatbot.compile_graph(checkpointer=self.__checkpointer)
        self.__chat_states[thread_id] = compiled_graph
        return compiled_graph
    

    def __tavily_search_tools(self) -> BaseTool:
        tool = TavilySearch(
            max_result=3, topic="general"
        )
        return tool
    
    def __tavily_extract_tool(self) -> BaseTool:
        return TavilyExtract()

    def get_vector_search_tool(self) -> BaseTool:
        @tool(
            name_or_callable="vector_search",
            args_schema=VectorSearchInput,
            description="Search for relevant documents in the vector database using semantic similarity."
        )
        def _vector_search(query: str, k: int = 3, score_threshold: float = 0.6) -> VectorSearchOutput:
            related_docs = self.__pgvector.similarity_search_with_relevance_scores(
                query=query, k=k, score_threshold=score_threshold
            )
            output = VectorSearchOutput(sources=set(), query=query, documents=[], scores=[], total_results=0)
            output.scores = list()
            output.sources = set()
            for doc, score in related_docs:
                output.documents.append(doc.page_content)
                output.scores.append(score)
                output.total_results += 1
                output.sources.add(doc.metadata.get("document_name", "unknown"))
            return output

        return _vector_search


    def __create_workflow_routing_prompt(self) -> RunnableSerializable:
        prompt_template = self.__prompt_loader.get_prompt("workflow_router")
        prompt = ChatPromptTemplate.from_template(prompt_template)
        return prompt | self.__generative_adapter.chat_model.with_structured_output(RouterOutputV2)
    
    def __create_workflow_routing_chain(self, state: StateV2, config: RunnableConfig) -> StateV2:
        file_name = state.get("file_name", "")
        if "csv" in file_name or "xlsx" in file_name:
            return {"router_output": "pandas_dataframe_agent", "question": state["messages"][-1], "db_name": "", "file_name": file_name, "dataframe_from_user": state.get("dataframe_from_user", None)}
        question = state["messages"][-1]
        conversation_memory = state.get("conversation_memory", "No conversation memory available")
        
        self.__logger.info(f"Workflow routing - Question: {question}")
        self.__logger.info(f"Workflow routing - Conversation memory: {conversation_memory[:100]}...")

        dataset_name = "" if state.get("database_config") is None or len(state.get("database_config")) == 0 else ", ".join([db.dataset_name for db in state["database_config"]])
        table_name = "" if state.get("database_config") is None or len(state.get("database_config")) == 0 else ", ".join([db.table_name for db in state["database_config"]])
        if dataset_name == "" and table_name == "":
            output: RouterOutputV2 = RouterOutputV2(
                output_router="specific_assistance",
                rephrased_question=question.content,
                table_name=""
            )
            return {"router_output": output.output_router, "question": output.rephrased_question, "db_name": output.table_name}
        response = self.__create_workflow_routing_prompt().invoke(
            {
                "input": question,
                "conversation_memory": conversation_memory,
                "dataset_name": dataset_name,
                "table_name": table_name
            },
            config
        )
        output_router: RouterOutputV2 = response
        self.__logger.info(f"Workflow routing result: {output_router.output_router}")
        
        return {"router_output": output_router.output_router, "question": output_router.rephrased_question, "db_name": output_router.table_name}
    
    def __llm_router(self, state: StateV2) -> Literal["specific_assistance", "sql_assistant", "pandas_dataframe_agent"]:
        route = state.get("router_output")
        self.__logger.info("llm router answer: {}".format(route))
        if route == "sql_assistant":
            return "sql_assistant"
        if route == "pandas_dataframe_agent":
            return "pandas_dataframe_agent"
        return "specific_assistance"    
        
    def __specific_knowledge_agent(self, conversation_memory: str = "") -> Runnable:
        system_prompt_template = self.__prompt_loader.get_prompt("specific_knowledge_agent")
        system_prompt = system_prompt_template.format(conversation_memory=conversation_memory)
        
        react_agent = create_react_agent(
            debug=True,            
            response_format=AgentResponseV2,
            model=self.__generative_adapter.chat_model, 
            tools=[self.get_vector_search_tool(), self.__tavily_search_tools(), self.__tavily_extract_tool()],
            prompt=system_prompt
        )
        return react_agent
    
    def __specific_knowledge_agent_chain(self, state: StateV2, config: RunnableConfig) -> StateV2:
        try:
            # Get conversation memory from state
            memory_context = state.get("conversation_memory", "No conversation memory available")
            self.__logger.info(f"Specific knowledge agent - Memory context: {memory_context[:100]}...")
            
            # Create agent with memory context
            agent = self.__specific_knowledge_agent(conversation_memory=memory_context)
            
            # Use the question from state
            question = state.get("question", "")
            if not question and state.get("messages"):
                question = state["messages"][-1].content
            
            self.__logger.info(f"Specific knowledge agent - Question: {question}")
            
            response = agent.invoke({"messages": [question]}, config=config)
            output: AgentResponseV2 = response.get("structured_response")
            self.__logger.info(f"Specific knowledge agent - Response generated: {output.answer[:100] if output and hasattr(output, 'answer') else 'No response'}...")
            
            return {"agent_answer": output}
        except Exception as e:
            self.__logger.error(f"Error in specific knowledge agent chain: {e}")
            print(e)
            raise
    
    def get_chatbot_v2(self, thread_id: str) -> CompiledStateGraph | None:
        try:
            compiled_graph = self.__chat_states.get(thread_id) 
            return compiled_graph
        except Exception:
            raise

    def __should_summarize(self, state: StateV2) -> Literal["llm_router", "initial_summary"]:
        has_docs = len(state.get("document_from_user", [])) > 0
        self.__logger.info(f"=== SHOULD SUMMARIZE DEBUG ===")
        self.__logger.info(f"Has documents: {has_docs}")
        self.__logger.info(f"Document count: {len(state.get('document_from_user', []))}")
        self.__logger.info(f"Routing to: {'initial_summary' if has_docs else 'llm_router'}")
        return "initial_summary" if has_docs else "llm_router"

    def __should_refine(self, state: StateV2) -> Literal["summary_refinement", "store_response"]:
        current_idx = state.get("document_idx", 0)
        total_docs = len(state.get("document_from_user", []))
        should_refine = current_idx < total_docs
        
        self.__logger.info(f"=== SHOULD REFINE DEBUG ===")
        self.__logger.info(f"Current document index: {current_idx}")
        self.__logger.info(f"Total documents: {total_docs}")
        self.__logger.info(f"Should refine: {should_refine}")
        self.__logger.info(f"Routing to: {'summary_refinement' if should_refine else 'store_response'}")
        
        return "summary_refinement" if should_refine else "store_response"

    def __create_initial_summary_chain(self):
        prompt = self.__prompt_loader.create_chat_prompt("initial_summary", role="human")
        return prompt | self.__generative_adapter.chat_model.with_structured_output(AgentResponseV2)
    
    def __generate_initial_summary(self, state: StateV2, config: RunnableConfig) -> StateV2:
        try:
            self.__logger.info("=== INITIAL SUMMARY DEBUG ===")
            self.__logger.info(f"Document content length: {len(state['document_from_user'][0].page_content)}")
            self.__logger.info(f"Document metadata: {state['document_from_user'][0].metadata}")
            self.__logger.info(f"Config: {config}")
            
            # Log the first 200 chars of document content
            doc_preview = state['document_from_user'][0].page_content[:200]
            self.__logger.info(f"Document preview: {doc_preview}...")
            
            answer = self.__create_initial_summary_chain().invoke(
                input={
                    "context": state["document_from_user"][0].page_content,
                },
                config=config,
            )
            
            self.__logger.info(f"Raw LLM response: {answer}")
            self.__logger.info(f"Answer length: {len(answer.answer) if hasattr(answer, 'answer') else 'No answer field'}")
            self.__logger.info(f"Answer content: {answer.answer[:300] if hasattr(answer, 'answer') else 'No answer field'}...")
            
            # Check for truncation patterns
            if hasattr(answer, 'answer') and answer.answer:
                if answer.answer.strip() in ["Berikut adalah ringkasan dari dokumen yang diberikan:", "Berikut adalah analisis", "Berdasarkan informasi yang tersedia"]:
                    self.__logger.warning("DETECTED TRUNCATION PATTERN in initial summary!")
                    self.__logger.warning(f"Truncated answer: '{answer.answer}'")
            
            answer.references.add(state["document_from_user"][0].metadata["document_name"])
            return {"agent_answer": answer, "document_idx": 1}
            
        except Exception as e:
            self.__logger.error(f"Error in initial summary generation: {e}")
            raise
    
    def __create_summary_refinement_chain(self):
        prompt = self.__prompt_loader.create_chat_prompt("summary_refinement", role="human")
        return prompt | self.__generative_adapter.chat_model.with_structured_output(AgentResponseV2)
    

    def __generate_summary_refinement(self, state: StateV2, config: RunnableConfig) -> StateV2:
        try:
            self.__logger.info("=== SUMMARY REFINEMENT DEBUG ===")
            self.__logger.info(f"Current document index: {state['document_idx']}")
            self.__logger.info(f"Total documents: {len(state['document_from_user'])}")
            self.__logger.info(f"Existing answer length: {len(state['agent_answer'].answer) if hasattr(state['agent_answer'], 'answer') else 'No answer field'}")
            self.__logger.info(f"Existing answer: {state['agent_answer'].answer[:200] if hasattr(state['agent_answer'], 'answer') else 'No answer field'}...")
            
            current_doc = state['document_from_user'][state['document_idx']]
            self.__logger.info(f"Current document content length: {len(current_doc.page_content)}")
            self.__logger.info(f"Current document metadata: {current_doc.metadata}")
            
            # Log the first 200 chars of current document content
            doc_preview = current_doc.page_content[:200]
            self.__logger.info(f"Current document preview: {doc_preview}...")
            
            refined = self.__create_summary_refinement_chain().invoke(
                {
                    "existing_answer": state["agent_answer"].answer,
                    "context": current_doc,
                },
                config=config,
            )
            
            self.__logger.info(f"Raw refined response: {refined}")
            self.__logger.info(f"Refined answer length: {len(refined.answer) if hasattr(refined, 'answer') else 'No answer field'}")
            self.__logger.info(f"Refined answer content: {refined.answer[:300] if hasattr(refined, 'answer') else 'No answer field'}...")
            
            # Check for truncation patterns
            if hasattr(refined, 'answer') and refined.answer:
                if refined.answer.strip() in ["Berikut adalah ringkasan dari dokumen yang diberikan:", "Berikut adalah analisis", "Berdasarkan informasi yang tersedia"]:
                    self.__logger.warning("DETECTED TRUNCATION PATTERN in summary refinement!")
                    self.__logger.warning(f"Truncated refined answer: '{refined.answer}'")
            
            refined.references.add(current_doc.metadata["document_name"])
            return {"agent_answer": refined, "document_idx": state["document_idx"] + 1}
            
        except Exception as e:
            self.__logger.error(f"Error in summary refinement: {e}")
            raise


    def __summarization_node(self) -> SummarizationNode:
        self.__logger.info("=== SUMMARIZATION NODE CREATED ===")
        self.__logger.info(f"Model: {self.__generative_adapter.chat_model}")
        self.__logger.info(f"Max tokens: 512")
        return SummarizationNode(
            model=self.__generative_adapter.chat_model,
            max_tokens=512,
        )

    def __memory_retrieval_node(self, state: StateV2, config: RunnableConfig) -> StateV2:
        """Retrieve relevant conversation memory based on current question"""
        try:
            current_question = state["messages"][-1].content if state.get("messages") else ""
            self.__logger.info(f"Memory retrieval for question: {current_question}")
            
            # Get conversation history from checkpointer
            thread_id = config.get("configurable", {}).get("thread_id", "default")
            self.__logger.info(f"Thread ID for memory retrieval: {thread_id}")
            
            if thread_id and hasattr(self.__checkpointer, 'get'):
                try:
                    # Retrieve conversation history
                    history_state = self.__checkpointer.get({"configurable": {"thread_id": thread_id}})
                    self.__logger.info(f"History state retrieved: {history_state is not None}")
                    messages = history_state.get("channel_values").get("messages")
                    if history_state and history_state.get("channel_values").get("messages"):
                        messages = history_state.get("channel_values").get("messages")
                        self.__logger.info(f"Found {len(messages)} messages in history")
                        
                        # Limit to last 10 messages to avoid context overflow
                        recent_history = messages[-10:] if len(messages) > 10 else messages
                        
                        # Create a simple memory summary using the LLM
                        memory_prompt_template = self.__prompt_loader.get_prompt("memory_retrieval")
                        memory_prompt = ChatPromptTemplate.from_template(memory_prompt_template)
                        
                        history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_history])
                        self.__logger.info(f"History text length: {len(history_text)}")
                        
                        memory_chain = memory_prompt | self.__generative_adapter.chat_model
                        memory_summary = memory_chain.invoke({
                            "question": current_question,
                            "history": history_text
                        })
                        
                        self.__logger.info(f"Memory summary generated: {memory_summary.content[:100]}...")
                        
                        return {
                            "conversation_memory": memory_summary.content,
                            "memory_retrieved": True
                        }
                    else:
                        self.__logger.info("No messages found in history state")
                        return {
                            "conversation_memory": "No previous conversation history",
                            "memory_retrieved": True
                        }
                        
                except Exception as e:
                    self.__logger.warning(f"Failed to retrieve conversation history: {e}")
                    return {
                        "conversation_memory": "No conversation memory available",
                        "memory_retrieved": False
                    }
            else:
                self.__logger.info(f"No thread_id or checkpointer not available. Thread_id: {thread_id}, Has checkpointer: {hasattr(self.__checkpointer, 'get')}")
                return {
                    "conversation_memory": "No conversation thread available",
                    "memory_retrieved": False
                }
            
        except Exception as e:
            self.__logger.error(f"Error in memory retrieval: {e}")
            return {
                "conversation_memory": "No conversation memory available",
                "memory_retrieved": False
            }

    def __memory_update_node(self, state: StateV2, config: RunnableConfig) -> StateV2:
        """Update conversation memory with current interaction"""
        try:
            current_memory = state.get("conversation_memory", "")
            current_answer = state.get("agent_answer", "")
            
            if not current_answer:
                return state
            
            # Create a simple memory update using the LLM
            update_prompt_template = self.__prompt_loader.get_prompt("memory_update")
            update_prompt = ChatPromptTemplate.from_template(update_prompt_template)
            
            # Generate updated memory
            update_chain = update_prompt | self.__generative_adapter.chat_model
            updated_memory = update_chain.invoke({
                "current_memory": current_memory,
                "question": state.get("question", ""),
                "answer": current_answer.answer if hasattr(current_answer, 'answer') else str(current_answer)
            })
            
            return {
                "conversation_memory": updated_memory.content,
                "memory_updated": True
            }
            
        except Exception as e:
            self.__logger.error(f"Error in memory update: {e}")
            return {
                "memory_updated": False
            }
        
    def __sql_agent(self, db: SQLDatabase, toolkit: SQLDatabaseToolkit, column_metadata: str, conversation_memory: str = "") -> Runnable:
        system_prompt_template = self.__prompt_loader.get_prompt("sql_agent")
        system_prompt = system_prompt_template.format(
            conversation_memory=conversation_memory,
            dialect=db.dialect,
            table_info=db.get_table_info(),
            usable_table_names=db.get_usable_table_names(),
            column_metadata=column_metadata
        )
        react_agent = create_react_agent(
            debug=True,            
            response_format=AgentResponseV2,
            model=self.__generative_adapter.chat_model, 
            tools=toolkit.get_tools(),
            prompt=system_prompt
        )
        return react_agent
        
    def __sql_agent_chain(self, state: StateV2, config: RunnableConfig) -> StateV2:
        try:
            db_config: List[DatabaseConfig] = state.get("database_config", [])
            db_name: str = state.get("db_name", "")
            
            if not db_config:
                return {"agent_answer": AgentResponseV2(
                    answer="No database configuration found. Please check your database settings.",
                    references=set(),
                    needs_clarification=False
                )}
            
            selected_db = [db for db in db_config if db.table_name == db_name]
            if not selected_db:
                return {"agent_answer": AgentResponseV2(
                    answer=f"Database '{db_name}' not found in available configurations.",
                    references=set(),
                    needs_clarification=False
                )}
            
            selected_db = selected_db[0]
            sql_db = self.__create_sql_database(str(selected_db.db_uri), {selected_db.table_name})
            toolkit = SQLDatabaseToolkit(db=sql_db, llm=self.__generative_adapter.chat_model)
            
            # Pass conversation memory to the agent
            conversation_memory = state.get("conversation_memory", "No conversation memory available")
            agent = self.__sql_agent(sql_db, toolkit, column_metadata=selected_db.column_metadata, conversation_memory=conversation_memory)
            
            # Create enhanced question with memory context
            memory_context = f"Context: {conversation_memory}\n\nQuestion: {state.get('question', '')}"
            
            # Use correct input format for create_react_agent
            response = agent.invoke({"messages": [memory_context]}, config=config)
            
            structured_response: AgentResponseV2 = response.get("structured_response")
            
            return {"agent_answer": structured_response}
            
        except Exception as e:
            return {"agent_answer": AgentResponseV2(
                answer=f"Error executing SQL query: {str(e)}",
                references=set(),
                needs_clarification=False
            )}

    def __chart_chain(self) -> Runnable:
        prompt_template = self.__prompt_loader.get_prompt("chart_generator")
        prompt = ChatPromptTemplate.from_template(prompt_template)
        return prompt | self.__generative_adapter.chat_model.with_structured_output(ChartData)
    
    def __generate_chart(self, state: StateV2, config: RunnableConfig) -> StateV2:
        agent_answer = state.get("agent_answer")
        chart = self.__chart_chain().invoke(
            {
                "answer": agent_answer.answer
            }, config=config)
        chartData: ChartData = chart
        agent_answer.chart = chartData
        return {"agent_answer": agent_answer}

    def __store_response(self, state: StateV2, config: RunnableConfig) -> StateV2:
        try:
            self.__logger.info("=== STORE RESPONSE DEBUG ===")
            agent_answer = state.get("agent_answer")
            
            if agent_answer:
                self.__logger.info(f"Agent answer type: {type(agent_answer)}")
                self.__logger.info(f"Agent answer: {agent_answer}")
                
                if hasattr(agent_answer, 'answer'):
                    self.__logger.info(f"Final answer length: {len(agent_answer.answer)}")
                    self.__logger.info(f"Final answer content: {agent_answer.answer}")
                    
                    # Check for truncation patterns in final output
                    if agent_answer.answer.strip() in ["Berikut adalah ringkasan dari dokumen yang diberikan:", "Berikut adalah analisis", "Berdasarkan informasi yang tersedia"]:
                        self.__logger.error("FINAL TRUNCATION DETECTED in store_response!")
                        self.__logger.error(f"Truncated final answer: '{agent_answer.answer}'")
                
                # Log the JSON that will be stored
                json_content = agent_answer.model_dump_json()
                self.__logger.info(f"JSON content length: {len(json_content)}")
                self.__logger.info(f"JSON content: {json_content}")
            else:
                self.__logger.warning("No agent_answer found in state")
            
            return {"messages": [AIMessage(content=agent_answer.model_dump_json())], "dataframe_from_user": None}
            
        except Exception as e:
            self.__logger.error(f"Error in store_response: {e}")
            raise


    @staticmethod
    def get_chat_history_v2(config: RunnableConfig, compiled_graph: CompiledStateGraph) -> List[ConversationStateV2]:
        """Return list of ConversationState objects for external rendering."""
        conversation_list: List[ConversationStateV2] = []
        conv_state = compiled_graph.get_state(config).values.get("messages") or []
        for idx, msg in enumerate(conv_state, start=1):
            if idx % 2:  # odd = question
                conv = ConversationStateV2()
                conv.question = msg.content
            else:  # even = answer
                agent_response: AgentResponseV2 = AgentResponseV2.model_validate_json(msg.content)
                conv.answer = agent_response
                conversation_list.append(conv)
        return conversation_list

    def __pandas_dataframe_chain(self, state: StateV2, config: RunnableConfig) -> StateV2:
        dataframe_from_user = state.get("dataframe_from_user")
        if dataframe_from_user is not None:
            dataframe_from_user = read_json(StringIO(dataframe_from_user))
        
        prefix_template = self.__prompt_loader.get_prompt("pandas_dataframe_agent")
        prefix = prefix_template.format(conversation_memory=state.get("conversation_memory"))

        agent = create_pandas_dataframe_agent(
            self.__generative_adapter.chat_model, 
            dataframe_from_user, 
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
            verbose=True,
            prefix=prefix,
            allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True})
        result = agent.invoke(state.get("question"), config=config).get("output")
        
        file_name = state.get("file_name")
        references = {file_name} if file_name else set()

        agent_response: AgentResponseV2 = AgentResponseV2(
            answer=result,
            references=references,
            needs_clarification=False
        )
        return {"agent_answer": agent_response, "dataframe_from_user": None}
