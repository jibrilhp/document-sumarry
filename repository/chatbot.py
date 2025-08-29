from typing import Dict, Literal, List
import re
import logging
import os

from entity.conversation import ConversationalChatbot, State, ConversationState, ConversationStateV2   
from infra.data_store import PostgresAdapter
from infra.generative_provider import GenerativeAdapter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END
from langgraph.graph.state import CompiledStateGraph

# SQL Agent imports
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine, text

from entity.document import ShapSummary, XAIFileSummary


def _is_speech_request(text: str) -> bool:
    """Very lightweight detector for 'write a presidential speech' prompts."""
    keywords = (
        "pidato",
        "buatkan pidato",
        "tuliskan pidato",
        "presidential speech",
        "speech for president",
    )
    t = text.lower()
    return any(k in t for k in keywords)


def _needs_sentiment_sim(text: str) -> bool:
    """
    Detects requests asking for Indonesian public sentiment
    or acceptance of a rule/policy.
    """
    keys = (
        "sentimen masyarakat",
        "sentimen publik",
        "public sentiment",
        "diterima masyarakat",
        "acceptance",
        "penerimaan",
    )
    t = text.lower()
    return any(k in t for k in keys)


def _needs_database_analysis(text: str) -> bool:
    """
    Detects requests for database analysis in Indonesian or English.
    """
    keywords = (
        # Indonesian patterns
        "analisis database",
        "analisa database",
        "isi database",
        "isi dari database",
        "database apa",
        "tabel database",
        "tabel apa saja",
        "analisis dari tabel"
        "struktur database",
        "skema database",
        "data dari database",
        "query database",
        "cari di database",
        "database berisi",
        "apa isi database",
        "tampilkan database",
        "lihat database",
        "database dtskul",  # specific for your case
        "dtskul database",

        # English patterns
        "analyze database",
        "database analysis",
        "database content",
        "what's in database",
        "database structure",
        "database schema",
        "show database",
        "list tables",
        "database tables",
        "table structure",
        "sql query",
        "query the database",
        "database report",
        "data from database",

        # Mixed patterns
        "database",  # if standalone, likely refers to DB
        "db content",
        "db structure"
    )

    t = text.lower()

    # Check for direct database name mentions
    db_indicators = ["database", "db", "tabel", "table", "data"]
    has_db_indicator = any(indicator in t for indicator in db_indicators)

    # Check for question words that might indicate database inquiry
    question_words = ["apa", "what", "bagaimana", "how", "berapa", "show", "tampilkan", "lihat"]
    has_question_word = any(word in t for word in question_words)

    # If it has both DB indicator and question word, likely a DB query
    if has_db_indicator and has_question_word:
        return True

    # Check against specific keywords
    return any(k in t for k in keywords)


def _is_xai_request(text: str) -> bool:
    """
    Detects requests for explaining anomalies from the provided CSV data.
    """
    keywords = (
        "jelaskan anomali",
        "explain anomaly",
        "mengapa anomali",
        "why is this an anomaly",
        "anomaly reason",
        "penyebab anomali",
        "explain the anomaly reason",
    )
    t = text.lower()
    return any(k in t for k in keywords)


class DatabaseConfig:
    """Configuration for database connections"""
    def __init__(self,
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "your_database",
                 username: str = "postgres",
                 password: str = "password"):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password

    def get_connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


# class SQLAgentManager:
#     """Manages SQL database connections and agent creation"""

#     def __init__(self, generative_adapter: GenerativeAdapter, db_config: DatabaseConfig):
#         self.generative_adapter = generative_adapter
#         self.db_config = db_config
#         self.db_connection = None
#         self.sql_agent = None
#         self.logger = logging.getLogger(__name__)
#         self.initialization_error = None

#     def initialize_sql_agent(self) -> bool:
#         """Initialize the SQL agent with PostgreSQL connection"""
#         self.logger.info("Starting SQL Agent initialization...")

#         try:
#             # Step 1: Test database connection
#             self.logger.info(f"Connecting to database: {self.db_config.database} at {self.db_config.host}:{self.db_config.port}")
#             connection_string = self.db_config.get_connection_string()

#             # Create engine with additional parameters for better connection handling
#             engine = create_engine(
#                 connection_string,
#                 pool_pre_ping=True,  # Verify connections before use
#                 pool_recycle=3600,   # Recycle connections after 1 hour
#                 echo=False           # Set to True for SQL debugging
#             )

#             # Test connection
#             with engine.connect() as conn:
#                 result = conn.execute(text("SELECT 1 as test"))
#                 test_value = result.fetchone()[0]
#                 if test_value != 1:
#                     raise Exception("Database connection test failed")

#             self.logger.info("✓ Database connection successful")

#             # Step 2: Create SQLDatabase wrapper
#             self.db_connection = SQLDatabase(engine=engine)
#             tables = self.db_connection.get_table_names()
#             self.logger.info(f"✓ Found {len(tables)} tables: {tables}")

#             if not tables:
#                 self.logger.warning("⚠️  No tables found in database. SQL analysis will be limited.")

#             # Step 3: Rebuild toolkit model
#             try:
#                 SQLDatabaseToolkit.model_rebuild()
#                 self.logger.info("✓ SQLDatabaseToolkit model rebuilt")
#             except Exception as e:
#                 self.logger.warning(f"⚠️  Could not rebuild SQLDatabaseToolkit model: {e}")

#             # Step 4: Verify LLM is available
#             if not hasattr(self.generative_adapter, 'chat_model') or self.generative_adapter.chat_model is None:
#                 raise Exception("Generative adapter chat_model is not available")

#             self.logger.info("✓ LLM model is available")

#             # Step 5: Create SQL agent with multiple fallback strategies
#             agent_configs = [
#                 # Primary configuration
#                 {
#                     "agent_type": "openai-tools",
#                     "verbose": True,
#                     "handle_parsing_errors": True,
#                     "max_iterations": 6,
#                     "max_execution_time": 60
#                 },
#                 # Fallback 1: Without execution time limit
#                 {
#                     "agent_type": "openai-tools",
#                     "verbose": True,
#                     "handle_parsing_errors": True,
#                     "max_iterations": 6
#                 },
#                 # Fallback 2: Default configuration
#                 {
#                     "verbose": True,
#                     "handle_parsing_errors": True
#                 },
#                 # Fallback 3: Minimal configuration
#                 {
#                     "verbose": False
#                 }
#             ]

#             for i, config in enumerate(agent_configs):
#                 try:
#                     self.logger.info(f"Trying SQL agent configuration {i+1}...")
#                     self.sql_agent = create_sql_agent(
#                         llm=self.generative_adapter.chat_model,
#                         db=self.db_connection,
#                         **config
#                     )

#                     # Test the agent with a simple query
#                     test_response = self.sql_agent.invoke({"input": "What tables are available in this database?"})
#                     if test_response:
#                         self.logger.info(f"✓ SQL Agent created successfully with configuration {i+1}")
#                         return True

#                 except Exception as e:
#                     self.logger.warning(f"Configuration {i+1} failed: {e}")
#                     continue

#             # If we reach here, all configurations failed
#             raise Exception("All SQL agent configurations failed")

#         except Exception as e:
#             error_msg = f"SQL Agent initialization failed: {str(e)}"
#             self.logger.error(error_msg)
#             self.initialization_error = error_msg
#             return False

#     def get_initialization_status(self) -> dict:
#         """Get detailed initialization status for debugging"""
#         return {
#             "sql_agent_initialized": self.sql_agent is not None,
#             "db_connection_initialized": self.db_connection is not None,
#             "initialization_error": self.initialization_error,
#             "database_config": {
#                 "host": self.db_config.host,
#                 "port": self.db_config.port,
#                 "database": self.db_config.database,
#                 "username": self.db_config.username
#             },
#             "available_tables": self.db_connection.get_table_names() if self.db_connection else None
#         }

#     def query_database(self, question: str) -> str:
#         """Execute database query using the SQL agent"""
#         if not self.sql_agent:
#             error_details = self.get_initialization_status()
#             error_msg = f"""
# ❌ **SQL Agent Tidak Tersedia**

# **Detail Error:**
# - SQL Agent: {'✓ Initialized' if error_details['sql_agent_initialized'] else '❌ Not Initialized'}
# - Database Connection: {'✓ Connected' if error_details['db_connection_initialized'] else '❌ Not Connected'}
# - Error: {error_details['initialization_error'] or 'Unknown error'}

# **Database Config:**
# - Host: {error_details['database_config']['host']}
# - Port: {error_details['database_config']['port']}
# - Database: {error_details['database_config']['database']}
# - Username: {error_details['database_config']['username']}

# **Troubleshooting:**
# 1. Pastikan database PostgreSQL berjalan
# 2. Periksa kredensial database di environment variables
# 3. Pastikan database dan tabel yang dianalisis ada
# 4. Periksa koneksi jaringan ke database

# Silakan hubungi administrator untuk mengatasi masalah ini.
#             """
#             return error_msg.strip()

#         try:
#             self.logger.info(f"Executing database query: {question[:100]}...")

#             # Use invoke method for newer LangChain versions
#             response = self.sql_agent.invoke({"input": question})
#             result = response.get("output", str(response))

#             self.logger.info("Database query executed successfully")
#             return result

#         except AttributeError:
#             # Fallback to run method
#             try:
#                 self.logger.info("Using fallback run method for SQL agent")
#                 response = self.sql_agent.run(question)
#                 return str(response)
#             except Exception as e:
#                 error_msg = f"Error menjalankan query database (fallback): {str(e)}"
#                 self.logger.error(error_msg)
#                 return error_msg
#         except Exception as e:
#             error_msg = f"Error dalam analisis database: {str(e)}"
#             self.logger.error(error_msg)
#             return error_msg

#     def get_database_summary(self) -> str:
#         """Get a summary of available database tables and structure"""
#         if not self.db_connection:
#             return "Database connection tidak tersedia. Silakan periksa konfigurasi database."

#         try:
#             tables = self.db_connection.get_table_names()
#             if not tables:
#                 return f"Database '{self.db_config.database}' terhubung, tetapi tidak ada tabel yang ditemukan."

#             summary = f"📊 **Database: {self.db_config.database}**\n"
#             summary += f"🔗 Host: {self.db_config.host}:{self.db_config.port}\n"
#             summary += f"📋 Jumlah tabel: {len(tables)}\n\n"

#             summary += "**Daftar Tabel dan Struktur:**\n"

#             for i, table in enumerate(tables, 1):
#                 try:
#                     # Get table info with sample data count
#                     table_info = self.db_connection.get_table_info([table])

#                     # Try to get row count
#                     try:
#                         with self.db_connection._engine.connect() as conn:
#                             count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
#                             row_count = count_result[0] if count_result else "Unknown"
#                     except:
#                         row_count = "Unknown"

#                     summary += f"\n**{i}. {table.upper()}** ({row_count} baris)\n"

#                     # Parse table info to show columns more cleanly
#                     if "CREATE TABLE" in table_info:
#                         # Extract column information
#                         lines = table_info.split('\n')
#                         columns = []
#                         for line in lines:
#                             if line.strip() and not line.strip().startswith('CREATE') and not line.strip().startswith(')'):
#                                 clean_line = line.strip().rstrip(',')
#                                 if clean_line and not clean_line.startswith('('):
#                                     columns.append(f"   • {clean_line}")

#                         if columns:
#                             summary += "   **Kolom:**\n"
#                             summary += '\n'.join(columns[:5])  # Show first 5 columns
#                             if len(columns) > 5:
#                                 summary += f"\n   • ... dan {len(columns) - 5} kolom lainnya"
#                     else:
#                         # Fallback: show raw table info (truncated)
#                         info_lines = table_info.split('\n')[:5]
#                         for line in info_lines:
#                             if line.strip():
#                                 summary += f"   {line.strip()}\n"

#                     summary += "\n"

#                 except Exception as e:
#                     summary += f"   ⚠️ Error getting table info: {str(e)}\n\n"

#             summary += "\n💡 **Tips:**\n"
#             summary += "• Tanyakan 'berapa baris di tabel [nama_tabel]' untuk mengetahui jumlah data\n"
#             summary += "• Tanyakan 'tampilkan 5 data pertama dari [nama_tabel]' untuk melihat sample data\n"
#             summary += "• Tanyakan 'struktur tabel [nama_tabel]' untuk detail kolom\n"

#             return summary

#         except Exception as e:
#             error_msg = f"Error mendapatkan informasi database: {str(e)}"
#             self.logger.error(error_msg)
#             return error_msg


class ChatBotRepository:
    def __init__(
        self,
        generative_provider: GenerativeAdapter,
        postgres_adapter: PostgresAdapter,
        db_config: DatabaseConfig = None
    ):
        self.chat_states: Dict[str, ConversationalChatbot] = {}
        self.checkpointer = postgres_adapter.get_checkpointer()
        self.generative_adapter = generative_provider
        self.pgvector = postgres_adapter.get_vector_store()
        self.logger = logging.getLogger(__name__)

        # Initialize SQL Agent Manager
        # if db_config is None:
            # Use environment variables or defaults
            # db_config = DatabaseConfig(
            #     host=os.getenv("ANALYST_DB_HOST", "localhost"),
            #     port=int(os.getenv("ANALYST_DB_PORT", "5432")),
            #     database=os.getenv("ANALYST_DB_NAME", "your_database"),
            #     username=os.getenv("ANALYST_DB_USER", "postgres"),
            #     password=os.getenv("ANALYST_DB_PASSWORD", "password")
            # )

        # self.sql_manager = SQLAgentManager(generative_provider, db_config)

        # Initialize SQL agent on startup
        # self.sql_initialization_success = self.sql_manager.initialize_sql_agent()
        # if not self.sql_initialization_success:
            # self.logger.warning("SQL Agent initialization failed. Database analysis features will be unavailable.")
            # Log detailed error information
            # status = self.sql_manager.get_initialization_status()
            # self.logger.error(f"SQL Agent Status: {status}")
        # else:
            # self.logger.info("SQL Agent initialized successfully")

    def get_sql_status(self) -> dict:
        """Get SQL agent status for debugging"""
        return self.sql_manager.get_initialization_status()

    def test_detection(self, text: str) -> dict:
        """Test which detection functions trigger for a given text"""
        return {
            "text": text,
            "needs_database_analysis": _needs_database_analysis(text),
            "needs_sentiment_sim": _needs_sentiment_sim(text),
            "is_speech_request": _is_speech_request(text),
            "is_xai_request": _is_xai_request(text),
            "routing_decision": self._get_routing_decision(text)
        }

    def _get_routing_decision(self, text: str) -> str:
        """Helper method to show what routing decision would be made"""
        if _is_xai_request(text):
            return "parse_anomaly_reason -> explain_anomaly"
        elif _needs_database_analysis(text):
            return "get_database_summary -> analyze_database"
        elif _needs_sentiment_sim(text):
            return "simulate_sentiment"
        else:
            return "generate_chat_response"

    def create_chatbot(self, thread_id: str) -> CompiledStateGraph:
        chatbot = ConversationalChatbot()

        # Existing nodes
        chatbot.add_node("generate_initial_summary", self.__generate_initial_summary)
        chatbot.add_node("refine_summary", self.__generate_summary_refinement)
        chatbot.add_node("fetch_context", self.__fetch_context)
        chatbot.add_node("simulate_sentiment", self.__simulate_sentiment)
        chatbot.add_node("generate_chat_response", self.__generate_chat_response)
        chatbot.add_node("add_answer_to_conversation", self.__add_answer_to_conversation)

        # SQL analysis nodes
        chatbot.add_node("analyze_database", self.__analyze_database)
        chatbot.add_node("get_database_summary", self.__get_database_summary)
        
        # XAI Anomaly Explanation nodes
        chatbot.add_node("parse_anomaly_reason", self.__parse_anomaly_reason)
        chatbot.add_node("explain_anomaly", self.__explain_anomaly)


        # Root → summarise or skip
        chatbot.add_conditional_edges(source=START, path=self.__should_summarize)

        # Summarisation refinement loop
        chatbot.add_conditional_edges(
            source="generate_initial_summary", path=self.__should_refine
        )
        chatbot.add_conditional_edges(
            source="refine_summary", path=self.__should_refine
        )

        # Enhanced routing after fetch_context
        chatbot.add_conditional_edges(
            source="fetch_context", path=self.__route_after_context
        )

        # Database analysis routing
        chatbot.add_conditional_edges(
            source="get_database_summary", path=self.__route_after_db_summary
        )
        
        # XAI analysis routing
        chatbot.add_edges("parse_anomaly_reason", "explain_anomaly")


        # End paths
        chatbot.add_edges("simulate_sentiment", "add_answer_to_conversation")
        chatbot.add_edges("generate_chat_response", "add_answer_to_conversation")
        chatbot.add_edges("analyze_database", "add_answer_to_conversation")
        chatbot.add_edges("explain_anomaly", "add_answer_to_conversation")
        chatbot.add_edges("add_answer_to_conversation", END)

        compiled_graph = chatbot.compile_graph(checkpointer=self.checkpointer)
        self.chat_states[thread_id] = compiled_graph
        return compiled_graph

    def get_chatbot(self, thread_id: str) -> CompiledStateGraph | None:
        return self.chat_states.get(thread_id)

    # Existing chain methods (unchanged)
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
        return prompt | self.generative_adapter.chat_model | StrOutputParser()

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
        return prompt | self.generative_adapter.chat_model | StrOutputParser()

    def __create_answer_chain(self):
        template = r"""
            <MAIN_PROMPT>
            Anda adalah pembantu yang handal untuk menjawab pertanyaan dan meringkas dokumen.

            ### Konteks referensi
            {context}

            ### Riwayat Percakapan
            {conversation}

            ### Pertanyaan sekarang
            {question}

            ### Aturan global
            1. Balas dengan bahasa yang sama yang digunakan pengguna, kecuali diminta sebaliknya.
            2. Jika pengguna meminta *pidato Presiden Republik Indonesia*, ikuti aturan **MODE PIDATO** di bawah.
            3. Jika pengguna meminta analisis database, sistem akan mengarahkan ke analisis SQL khusus.
            4. Jika pengguna meminta penjelasan anomali, sistem akan mengarahkan ke alur kerja XAI.
            </MAIN_PROMPT>

            <MODE PIDATO>
            • Jika `_is_speech_request(question)` True:
            – Tanyakan info acara bila belum ada.
            – Setelah info lengkap, susun pidato resmi (salut, sapaan, pengantar, isi inti, penutup, salam).
            – Panjang ±600‑800 kata kecuali diminta lain.
            – Keluarkan **hanya teks pidato**.
            </MODE PIDATO>

            <EXAMPLES>
            ● *Speech request*
            User: "Buatkan pidato Presiden tentang Hari Lingkungan Hidup sedunia."
            Assistant: "Apakah pidato ini akan disampaikan di Istana Negara atau lokasi lain? …"

            ● *Sentiment simulation*
            User: "Bagaimana sentimen masyarakat Indonesia tentang larangan total plastik sekali pakai?"
            Assistant (di node lain) ➜
            • 65 % dukungan…
            _Disclaimer: ini simulasi AI berdasarkan data daring terbatas._

            ● *Database analysis*
            User: "Analisis database dari tabel penjualan"
            Assistant: (Akan diarahkan ke sistem analisis SQL khusus)
            
            ● *XAI Anomaly Explanation*
            User: "Jelaskan mengapa baris ini anomali"
            Assistant: (Akan diarahkan ke alur kerja XAI untuk memberikan penjelasan kontekstual)
            </EXAMPLES>
            """
        return (
            ChatPromptTemplate([("human", template)])
            | self.generative_adapter.chat_model
            | StrOutputParser()
        )

    # Existing node methods (unchanged)
    def __generate_initial_summary(self, state: State, config: RunnableConfig) -> State:
        summary = self.__create_initial_summary_chain().invoke(
            input=state["document_from_user"][0],
            config=config,
        )
        return {"answer": summary, "index": 1}

    def __generate_summary_refinement(self, state: State, config: RunnableConfig) -> State:
        refined = self.__create_summary_refinement_chain().invoke(
            {
                "existing_answer": state["answer"],
                "context": state["document_from_user"][state["index"]],
            },
            config=config,
        )
        return {"answer": refined, "index": state["index"] + 1}

    def __fetch_context(self, state: State) -> State:
        """Retrieve vector‑similar docs to the user question."""
        q_text = state["conversation"][-1].content
        vec_docs = self.pgvector.similarity_search_with_relevance_scores(
            query=q_text, score_threshold=0.6
        )
        context = "\n".join(doc.page_content for doc, _ in vec_docs)
        return {"context": context}

    def __simulate_sentiment(self, state: State) -> State:
        prompt = ChatPromptTemplate(
            [
                (
                    "human",
                    """\
                    Anda adalah analis kebijakan publik.

                    Tugas:
                    1. Jelaskan sentimen masyarakat Indonesia terhadap kebijakan/wacana berikut:
                    "{question}"
                    2. Rangkai pro & kontra utama, sertakan judul + media (tanpa URL).
                    3. Estimasikan tingkat penerimaan publik (%) + justifikasi ≤ 40 kata.
                    4. Tutup dengan *disclaimer* bahwa ini simulasi AI berbasis data daring terbatas.

                    Balas ≤ 250 kata, bahasa sama dengan pertanyaan.""",
                )
            ]
        )
        chain = prompt | self.generative_adapter.chat_model | StrOutputParser()
        answer = chain.invoke({"question": state["conversation"][-1].content})
        return {"answer": answer}

    def __generate_chat_response(self, state: State) -> State:
        chain = self.__create_answer_chain()
        answer = chain.invoke(
            {
                "conversation": state["conversation"],
                "question": state["conversation"][-1],
                "context": state["context"],
            }
        )
        return {"answer": answer}

    # SQL analysis node methods
    def __get_database_summary(self, state: State) -> State:
        """Get database summary and structure information"""
        question = state["conversation"][-1].content

        # Check if user is asking about a specific database
        if "dtskul" in question.lower() or "database" in question.lower():
            summary = self.sql_manager.get_database_summary()

            # Add context about the specific database request
            if "dtskul" in question.lower():
                summary = f"Database 'dtskul' Information:\n\n{summary}"
        else:
            summary = self.sql_manager.get_database_summary()

        return {"db_summary": summary}

    def __analyze_database(self, state: State) -> State:
        """Perform database analysis using SQL agent"""
        question = state["conversation"][-1].content

        # Check if we have database summary context
        context = state.get("db_summary", "")

        # Enhance the question with context if available
        if context:
            enhanced_question = f"""
            Berdasarkan struktur database berikut:
            {context}

            Pertanyaan pengguna: {question}

            Silakan analisis dan berikan jawaban yang komprehensif dengan query SQL yang sesuai.
            Buat output dalam bentuk tabel jika memungkinkan
            """
        else:
            enhanced_question = question

        answer = self.sql_manager.query_database(enhanced_question)
        return {"answer": answer}
        
    # XAI Anomaly Explanation Node Methods
    def __parse_anomaly_reason(self, state: State) -> State:
        """
        Parses the complex anomaly_reason string into a structured dictionary.
        This node should be preceded by a step that loads the anomaly data into the state.
        """
        # In a real app, 'raw_anomaly_reason' would be in the state.
        # We simulate it here for demonstration.
        raw_anomaly_reason = state.get("raw_anomaly_reason", "REASON NOT FOUND")

        parsed_reasons = []
        # Split by semicolon for multiple reasons
        reasons = raw_anomaly_reason.split(';')
        for reason in reasons:
            if not reason.strip():
                continue

            # Regex to capture the main parts
            match = re.match(r"([\w↔]+)\s+([\w\s]+)—?\s*(.*)", reason.strip())
            
            if match:
                relationship, reason_type, details = match.groups()
                parsed_reasons.append({
                    "relationship": relationship.strip(),
                    "reason_type": reason_type.strip(),
                    "details": details.strip()
                })
            else:
                # Fallback for reasons that don't match the primary pattern
                parsed_reasons.append({
                    "relationship": "General",
                    "reason_type": "Info",
                    "details": reason.strip()
                })

        return {"parsed_anomaly": parsed_reasons}

    def __explain_anomaly(self, state: State) -> State:
        """
        Generates a user-friendly explanation of the anomaly using its context.
        """
        parsed_anomaly = state.get("parsed_anomaly")
        anomaly_context = state.get("anomaly_context", {}) # Default to empty dict

        prompt = ChatPromptTemplate.from_template(
            """
            Anda adalah seorang analis data ahli dalam Explainable AI (XAI).
            Tugas Anda adalah menjelaskan alasan anomali dalam format yang mudah dimengerti.

            **Konteks Anomali:**
            - Nama Unit Kerja: {UNITKERJA_NAMA}
            - Nama Kegiatan: {KEGIATAN_NAMA}
            - Nama Sub-Kegiatan: {SUBKEG_NAMA}
            - Nama Rekening: {REKENING_NAMA}
            - Nama Komponen: {KOMPONEN_NAMA}

            **Alasan Anomali yang Terdeteksi (setelah di-parse):**
            {parsed_anomaly}

            **Instruksi Anda:**
            1.  Gunakan **Konteks Anomali** untuk membuat penjelasan Anda spesifik dan relevan.
            2.  Jelaskan setiap alasan anomali satu per satu dengan bahasa yang sederhana.
            3.  Fokus pada hubungan antar bidang yang menyebabkan masalah.
            4.  Tutup dengan rekomendasi singkat tentang apa yang harus diperiksa.

            **Contoh Jawaban:**
            "Ditemukan anomali pada data untuk komponen **[Nama Komponen]** dalam kegiatan **[Nama Kegiatan]**.
            Penyebabnya adalah sebagai berikut:
            -   **Kombinasi Rekening dan Sub-Kegiatan Tidak Wajar**: Sistem mendeteksi hubungan yang belum pernah ada sebelumnya antara Rekening '{REKENING_NAMA}' dan Sub-Kegiatan '{SUBKEG_NAMA}'.
            -   **Kombinasi Sub-Kegiatan dan Kegiatan Tidak Wajar**: Demikian pula, hubungan antara Sub-Kegiatan '{SUBKEG_NAMA}' dan Kegiatan '{KEGIATAN_NAMA}' juga merupakan kombinasi baru.

            **Rekomendasi:**
            Mohon periksa kembali kebenaran input data untuk memastikan bahwa kegiatan, sub-kegiatan, dan rekening yang dipilih sudah sesuai satu sama lain."

            Sekarang, berikan penjelasan untuk data yang diberikan.
            """
        )

        chain = prompt | self.generative_adapter.chat_model | StrOutputParser()
        
        # Prepare context, providing fallbacks for missing keys
        context_data = {
            "UNITKERJA_NAMA": anomaly_context.get("UNITKERJA_NAMA", "N/A"),
            "KEGIATAN_NAMA": anomaly_context.get("KEGIATAN_NAMA", "N/A"),
            "SUBKEG_NAMA": anomaly_context.get("SUBKEG_NAMA", "N/A"),
            "REKENING_NAMA": anomaly_context.get("REKENING_NAMA", "N/A"),
            "KOMPONEN_NAMA": anomaly_context.get("KOMPONEN_NAMA", "N/A"),
            "parsed_anomaly": parsed_anomaly
        }

        explanation = chain.invoke(context_data)

        return {"answer": explanation}


    def __add_answer_to_conversation(self, state: State) -> State:
        return {"conversation": [state["answer"]]}

    # Enhanced conditional edge methods
    def __should_summarize(self, state: State) -> Literal["fetch_context", "generate_initial_summary"]:
        has_docs = len(state.get("document_from_user", [])) > 0
        return "generate_initial_summary" if has_docs else "fetch_context"

    def __should_refine(self, state: State) -> Literal["refine_summary", "add_answer_to_conversation"]:
        return (
            "add_answer_to_conversation"
            if state["index"] >= len(state["document_from_user"])
            else "refine_summary"
        )

    def __route_after_context(
        self, state: State
    ) -> Literal["simulate_sentiment", "generate_chat_response", "get_database_summary", "parse_anomaly_reason"]:
        question = state["conversation"][-1].content

        if _is_xai_request(question):
            return "parse_anomaly_reason"
        elif _needs_database_analysis(question):
            return "get_database_summary"
        elif _needs_sentiment_sim(question):
            return "simulate_sentiment"
        else:
            return "generate_chat_response"

    def __route_after_db_summary(
        self, state: State
    ) -> Literal["analyze_database"]:
        """Always proceed to database analysis after getting summary"""
        return "analyze_database"

    @staticmethod
    def get_chat_history(config: RunnableConfig, compiled_graph: CompiledStateGraph) -> List[ConversationState]:
        """Return list of ConversationState objects for external rendering."""
        conversation_list: List[ConversationState] = []
        conv_state = compiled_graph.get_state(config).values.get("conversation") or []
        for idx, msg in enumerate(conv_state, start=1):
            if idx % 2:  # odd = question
                conv = ConversationState()
                conv.question = msg.content
            else:  # even = answer
                conv.answer = msg.content
                conversation_list.append(conv)
        return conversation_list

    def upload_xai_file(self, shap_summary: ShapSummary):
        prompt = ChatPromptTemplate.from_template(
            """
                You are given a dataset of SHAP values that explain an AI model's predictions. The dataset includes:

                - Number of rows (total predictions)

                - Number of features (factors considered by the model)

                - Distribution of predictions (e.g., Approved, Need Revision, Rejected)

                - Top features with their importance values

                Your task: Create a plain-language summary report for non-technical readers.
                The report should include: 

                - Introduction: Briefly describe what the AI model is predicting.

                - Prediction Results: Show how many items fall into each prediction category, in simple terms.

                - Key Influencing Factors: List the top factors that most influence the AI's decisions, explained in everyday language (e.g., “Project Cost” instead of “shap_cost_midr”).

                - Interpretation: Explain what these results mean in practical terms — what levers matter most, whether the model is balanced, and any high-level takeaways.

                - Tone: Write in a clear, neutral, and business-friendly style. Avoid technical jargon.

                The dataset is as follows:
                {shap_summary}

                The output should be in the following format:
                {{
                    "summary": "summary of the dataset",
                    "source": "source of the dataset"
                }}

                Answer in Bahasa Indonesia
            """
        )
        chain = prompt | self.generative_adapter.chat_model.with_structured_output(XAIFileSummary)
        response = chain.invoke({"shap_summary": shap_summary.model_dump_json()})
        return response

# Example usage and configuration
def create_chatbot_with_sql_support(
    generative_provider: GenerativeAdapter,
    postgres_adapter: PostgresAdapter,
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "your_database",
    db_user: str = "postgres",
    db_password: str = "password"
) -> ChatBotRepository:
    """
    Factory function to create a chatbot with SQL analysis support

    Args:
        generative_provider: Your existing generative model adapter
        postgres_adapter: Your existing PostgreSQL adapter
        db_host: Database host (default: localhost)
        db_port: Database port (default: 5432)
        db_name: Database name
        db_user: Database username
        db_password: Database password

    Returns:
        ChatBotRepository: Enhanced chatbot with SQL analysis capabilities
    """
    db_config = DatabaseConfig(
        host=db_host,
        port=db_port,
        database=db_name,
        username=db_user,
        password=db_password
    )

    return ChatBotRepository(
        generative_provider=generative_provider,
        postgres_adapter=postgres_adapter,
        db_config=db_config
    )