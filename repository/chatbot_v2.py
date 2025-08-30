from typing import Dict, Literal, List
from entity.conversation import ConversationalChatbot, ConversationStateV2
from entity.tools import VectorSearchInput, VectorSearchOutput
from infra.generative_provider import GenerativeAdapter
from infra.data_store import PostgresAdapter
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
import json

class ChatBotV2Repository:
    def __init__(
        self,
        generative_provider: GenerativeAdapter,
        postgres_adapter: PostgresAdapter,
    ):
        self.__generative_adapter = generative_provider
        self.__pgvector = postgres_adapter.get_vector_store()
        self.__checkpointer = postgres_adapter.get_checkpointer()
        self.__logger = logging.getLogger(__name__)
        self.__chat_states: Dict[str, CompiledStateGraph] = {}
        self.__create_sql_database = postgres_adapter.create_sql_database

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
        chatbot.add_edges("memory_update", "store_response")
        chatbot.add_edges("store_response", END)
        compiled_graph = chatbot.compile_graph(checkpointer=self.__checkpointer)
        g = compiled_graph.get_graph().draw_mermaid_png()
        with open("chatbot_v2.png", "w+b") as f:
            f.write(g)
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
        prompt = ChatPromptTemplate.from_template(
            """
                You are a workflow router for a LLM product.
                You should decide where to route user's request based on conversation and user question.
                
                Conversation Memory: {conversation_memory}
                
                Given user question and chat conversation you should route user request to either 2 workflows:
                
                1. specific_assistance:
                If user ask about specific knowledge and has correlation to previous conversation, use this workflow.
                2. sql_assistant:
                If user ask about either of these {dataset_name} related topics, use this workflow. Fill table_name of the response with one of these table names:{table_name}.
           
                You should output either `specific_assistance` and `sql_assistant`.

                Question:
                {input}
             """
        )
        return prompt | self.__generative_adapter.chat_model.with_structured_output(RouterOutputV2)
    
    def __create_workflow_routing_chain(self, state: StateV2, config: RunnableConfig) -> StateV2:
        question = state["messages"][-1]
        conversation_memory = state.get("conversation_memory", "No conversation memory available")
        
        self.__logger.info(f"Workflow routing - Question: {question}")
        self.__logger.info(f"Workflow routing - Conversation memory: {conversation_memory[:100]}...")
        
        response = self.__create_workflow_routing_prompt().invoke(
            {
                "input": question,
                "conversation_memory": conversation_memory,
                "dataset_name": ", ".join([db.dataset_name for db in state["database_config"]]),
                "table_name": ", ".join([db.table_name for db in state["database_config"]])
            },
            config
        )
        output_router: RouterOutputV2 = response
        self.__logger.info(f"Workflow routing result: {output_router.output_router}")
        
        return {"router_output": output_router.output_router, "question": output_router.rephrased_question, "db_name": output_router.table_name}
    
    def __llm_router(self, state: StateV2) -> Literal["specific_assistance", "sql_assistant"]:
        route = state.get("router_output")
        self.__logger.info("llm router answer: {}".format(route))
        if route == "sql_assistant":
            return "sql_assistant"
        return "specific_assistance"    
        
    def __specific_knowledge_agent(self, conversation_memory: str = "") -> Runnable:
        # Create a dynamic prompt that includes conversation memory
        system_prompt = f"""
            Kamu adalah asisten riset yang dapat menggabungkan informasi dari sumber internal (vector_search) dan eksternal (tavily_search, tavily_extract).

            Context dari percakapan sebelumnya: {conversation_memory}

            Aturan kerja:
            1. SELALU mulai dengan vector_search menggunakan pertanyaan pengguna.
            2. Evaluasi apakah hasil vector_search cukup untuk menjawab pertanyaan dengan yakin:
            - Cukup = mengandung fakta atau data relevan yang langsung menjawab pertanyaan.
            - Kurang = informasi tidak ada, tidak spesifik, atau tidak lengkap.
            3. Jika kurang, gunakan tavily_search untuk mencari informasi tambahan di web.
            4. Jika dari tavily_search terdapat URL yang relevan namun perlu isi lebih detail, gunakan tavily_extract.
            3. Jika kurang, gunakan tavily_search dan tavily_extract untuk mencari informasi tambahan di web
            5. Gabungkan semua informasi yang ditemukan menjadi satu jawaban terpadu.
            6. Ambil field `sources` dari `vector_search` untuk mengisi field `references`
            7. Jika setelah semua langkah jawaban masih belum lengkap atau ambigu, set `needs_clarification=true` dan jelaskan kekurangannya.
            8. Jika yakin, set `needs_clarification=false`.
            9. Gunakan context dari percakapan sebelumnya untuk memberikan jawaban yang lebih relevan dan kontekstual.
        """
        
        react_agent = create_react_agent(
            debug=False,            
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
        return "initial_summary" if has_docs else "llm_router"

    def __should_refine(self, state: StateV2) -> Literal["summary_refinement", "store_response"]:
        return "summary_refinement" if state.get("document_idx") < len(state.get("document_from_user")) else "store_response"

    def __create_initial_summary_chain(self):
        prompt = ChatPromptTemplate(
            [
                (
                    "human",
                    """\
                    Buatkan ringkasan terstruktur dari konteks berikut.

                    ### Perintah ringkasan
                    1. Maksimum 120 kata, bahasa sama dengan dokumen.
                    2. Gunakan poin‑poin (•) untuk fakta kunci.
                    3. Tutup dengan "👉 Ringkasan selesai".

                    {context}""",
                )
            ]
        )
        return prompt | self.__generative_adapter.chat_model.with_structured_output(AgentResponseV2)
    
    def __generate_initial_summary(self, state: StateV2, config: RunnableConfig) -> StateV2:
        answer = self.__create_initial_summary_chain().invoke(
            input=state["document_from_user"][0],
            config=config,
        )
        answer.references.add(state["document_from_user"][0].metadata["document_name"])
        return {"agent_answer": answer, "document_idx": 1}
    
    def __create_summary_refinement_chain(self):
        prompt = ChatPromptTemplate(
            [
                (
                    "human",
                    """\
                    Anda diminta *memperbaiki* ringkasan.

                    Ringkasan sementara:
                    {existing_answer}

                    Konteks tambahan:
                    {context}

                    ### Instruksi
                    • Tambahkan fakta penting yang belum tercakup.
                    • Perbaiki kesalahan, hindari pengulangan.
                    • Tetap ≤ 120 kata dan tutup dengan "👉 Ringkasan selesai".\
                    """,
                )
            ]
        )
        return prompt | self.__generative_adapter.chat_model.with_structured_output(AgentResponseV2)
    

    def __generate_summary_refinement(self, state: StateV2, config: RunnableConfig) -> StateV2:
        refined = self.__create_summary_refinement_chain().invoke(
            {
                "existing_answer": state["agent_answer"].answer,
                "context": state["document_from_user"][state["document_idx"]],
            },
            config=config,
        )
        refined.references.add(state["document_from_user"][state["document_idx"]].metadata["document_name"])
        return {"agent_answer": refined, "document_idx": state["document_idx"] + 1}


    def __summarization_node(self) -> SummarizationNode:
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
                        memory_prompt = ChatPromptTemplate.from_template("""
                            Based on the current user question, summarize the most relevant information from the conversation history in 2-3 sentences.
                            
                            Current Question: {question}
                            
                            Recent Conversation History:
                            {history}
                            
                            Relevant Summary:
                        """)
                        
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
            update_prompt = ChatPromptTemplate.from_template("""
                Update the conversation memory with the new information from this interaction.
                
                Current Memory: {current_memory}
                
                New Information:
                - Question: {question}
                - Answer: {answer}
                
                Instructions:
                1. Integrate the new information with existing memory
                2. Maintain important context and facts
                3. Keep the summary concise (3-4 sentences max)
                4. Focus on information that would be useful for future questions
                
                Updated Memory:
            """)
            
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
        system_prompt = f"""
            You are an agent designed to interact with a SQL database and provide data visualizations.

            Conversation Context: {conversation_memory}

            Important: You must fill *all relevant fields* of the AgentResponseV2 schema.
            Do not omit "chart" when data is visualizable.
            
            Important: "chart_spec" must be a valid JSON string. 
            Do not enclose it in quotes. Output raw JSON for the chart_spec field.

            Given an input question:
            1. Consider the conversation context to understand what the user is asking about
            2. Create a syntactically correct {db.dialect} query to run.
            3. Look at the results of the query and return the answer in JSON.

            Query guidelines:
            - Unless the user specifies a specific number of examples, always LIMIT results to at most 5.
            - You can order the results by a relevant column to return the most interesting examples.
            - Never query for all the columns from a table; only use relevant columns.
            - Double check your query before executing it.
            - If a query fails, rewrite and retry.
            - DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.).
            - Always check available tables first, then query the schema of the most relevant ones.

            Chart generation guidelines:
            - When the query returns numerical or categorical data that could benefit from visualization, generate a chart specification.
            - Use these defaults:
            - Time-series data → line chart
            - Categorical comparisons → bar chart
            - Correlation between two numerical variables → scatter plot
            - Proportions/percentages → pie chart
            - Only suggest a chart when data is meaningful and has at least 2 data points.
            - Always include the raw query results in the response.

            You MUST always return output that conforms to the AgentResponseV2 schema.

            Column Metadata:
            {column_metadata}

            - "answer": natural language explanation
            - "references": set of URLs (empty if none)
            - "needs_clarification": boolean
            - "chart": 
            - If data is visualizable, always fill with:
                - "data": SQL rows
                - "chart_spec": valid Vega-Lite v5 spec
            - If not visualizable, return null

            Do not place "data" and "chart_spec" at the top level. They must always be inside "chart".
        """
        react_agent = create_react_agent(
            debug=False,            
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
        prompt = ChatPromptTemplate.from_template("""
            You are a chart generator.  
            Given only the natural language answer from the database query, generate a valid Vega-Lite v5 chart specification.  

            RULES:  
            - Always return a JSON object with the following schema:
            {{
            "chart": {{
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "data": {{"values": [ ... ]}},
                "mark": "bar"
            }}
            }}

            Always return `chart_spec` as a valid JSON object (not a string).
            Do not wrap the chart specification in quotes.

            - Parse the entities and numbers from the answer into "data.values".
            - Select chart type:
            - categorical comparisons → "mark": "bar"
            - time series (date, year, month) → "mark": "line"
            - two quantitative variables → "mark": "point"
            - proportions/percentages → "mark": "arc"
            - Use meaningful axis titles from the answer.
            - Do not include explanations or extra text, only the JSON.  

            ANSWER:  
            {answer}
        """)
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
        agent_answer = state.get("agent_answer")
        return {"messages": [AIMessage(content=agent_answer.model_dump_json())]}


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

