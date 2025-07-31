from typing import Dict, Literal, List
from entity.conversation import ConversationalChatbot
from infra.generative_provider import GenerativeAdapter
from infra.data_store import PostgresAdapter
import logging
from langgraph.graph import START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.prompts import ChatPromptTemplate
from entity.conversation import StateV2, AgentResponseV2
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import RunnableSerializable
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import Runnable
from langchain_core.tools.base import BaseTool
from langchain_tavily import TavilySearch, TavilyExtract
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

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
        self.__agent_response_parser = PydanticOutputParser(pydantic_object=AgentResponseV2)

    def __tavily_search_tools(self) -> BaseTool:
        tool = TavilySearch(
            max_result=3, topic="general"
        )
        return tool
    
    def __tavily_extract_tool(self) -> BaseTool:
        return TavilyExtract()


    def __create_workflow_routing_prompt(self) -> RunnableSerializable:
        prompt = ChatPromptTemplate.from_template(
            """
                You are a workflow router for a LLM product.
                You should decide where to route user's request based on conversation and user question.
                Given user question  and chat conversation you should route user request to either 3 workflow
                
                1. general_assistance: 
                If user ask about general knowledge and has no correlation to previous conversation. You should route to this workflow
                2. specific_assistance:
                If user ask about specific knowledge and has correlation to previous conversation, use this workflow.
           
                You should output either `general_assistance` and `specific_assistance`.

                Question:
                {input}
             """
        )
        return prompt | self.__generative_adapter.chat_model
    
    def __create_workflow_routing_chain(self, state: StateV2, config: RunnableConfig) -> StateV2:
        question = state["messages"][-1]      
        response = self.__create_workflow_routing_prompt().invoke(
            {
                "input": question
            },
            config
        )
        output_router = str(response.content)
        output_router_msg = AIMessage(
            content=output_router
        )
        return {"messages": [output_router_msg], "router_output": output_router, "question": str(question)}
    
    def __llm_router(self, state: StateV2) -> Literal["specific_assistance", "general_assistance"]:
        route = state.get("router_output")
        self.__logger.info("llm router answer: {}".format(route))
        if route == "general_assistance":
            return "general_assistance"
        return "specific_assistance"    
    
    def __general_knowledge_agent(self) -> Runnable:
        react_agent = create_react_agent(
            debug=True,            
            response_format=AgentResponseV2,
            model=self.__generative_adapter.chat_model, 
            tools=[self.__tavily_search_tools(), self.__tavily_extract_tool()],
            prompt="""
             You are a friendly and knowledgeable assistant designed to help everyday users with clear, accurate, and engaging answers. Your mission is to answer questions in a way that's easy to understand, using a step-by-step process to ensure the best response. Here's how you work:

             1. **Get the Question**: Carefully read the user's question to understand what they're asking and why. Check the conversation history to see if past chats give extra context.
             2. **Gather Information**: If the question needs up-to-date info or details you don't have, use the web search tool to find relevant data. Summarize what you find to keep it focused and useful.
             3. **Make a Plan**: Combine your built-in knowledge, past conversation context, and any web search results to decide how to answer. Plan steps like answering directly, summarizing web info, or asking for clarification if the question isn't clear.
             4. **Deliver the Answer**: Follow your plan to create a response that's concise, friendly, and helpful

             **How to Behave**:
             - Keep answers simple, warm, and easy to follow, like chatting with a friend.
             - Only use the web search tool when you need fresh or specific info—don't overdo it.
             - If the question is vague, politely ask for more details to nail the response.
             - For tricky or sensitive topics, stay neutral and kind.
             - If you're unsure or can't find enough info, say so honestly and point users to reliable sources (e.g., “For more details, check trusted websites or official sources”).        
            """
        )
        return react_agent
    
    def __general_knowledge_agent_chain(self, state: StateV2, config: RunnableConfig) -> StateV2:
        try:
            agent = self.__specific_knowledge_agent()
            response = agent.invoke({"messages": [state.get("question")]}, config=config)
            output: AgentResponseV2 = response.get("structured_response")
            return {"agent_answer": output}
        except Exception as e:
            print(e)
            raise
            
        return state
    
        
    def __specific_knowledge_agent(self) -> Runnable:
        react_agent = create_react_agent(
            debug=True,            
            response_format=AgentResponseV2,
            model=self.__generative_adapter.chat_model, 
            tools=[self.__tavily_search_tools(), self.__tavily_extract_tool()],
            prompt="""
             You are a friendly and knowledgeable assistant designed to help everyday users with clear, accurate, and engaging answers. Your mission is to answer questions in a way that's easy to understand, using a step-by-step process to ensure the best response. Here's how you work:

             1. **Get the Question**: Carefully read the user's question to understand what they're asking and why. Check the conversation history to see if past chats give extra context.
             2. **Gather Information**: If the question needs up-to-date info or details you don't have, use the web search tool to find relevant data. Summarize what you find to keep it focused and useful.
             3. **Make a Plan**: Combine your built-in knowledge, past conversation context, and any web search results to decide how to answer. Plan steps like answering directly, summarizing web info, or asking for clarification if the question isn't clear.
             4. **Deliver the Answer**: Follow your plan to create a response that's concise, friendly, and helpful

             **How to Behave**:
             - Keep answers simple, warm, and easy to follow, like chatting with a friend.
             - Only use the web search tool when you need fresh or specific info—don't overdo it.
             - If the question is vague, politely ask for more details to nail the response.
             - For tricky or sensitive topics, stay neutral and kind.
             - If you're unsure or can't find enough info, say so honestly and point users to reliable sources (e.g., “For more details, check trusted websites or official sources”).        
            """
        )
        return react_agent
    
    def __specific_knowledge_agent_chain(self, state: StateV2, config: RunnableConfig) -> StateV2:
        try:
            agent = self.__specific_knowledge_agent()
            response = agent.invoke({"messages": [state.get("question")]}, config=config)
            output: AgentResponseV2 = response.get("structured_response")
            return {"agent_answer": output}
        except Exception as e:
            print(e)
            raise
            
        return state
    


    def create_chatbot_v2(self, thread_id: str):
        chatbot = ConversationalChatbot()
        chatbot.add_edges(START, "llm_router")
        chatbot.add_node("llm_router", self.__create_workflow_routing_chain)
        chatbot.add_node("general_assistance", self.__general_knowledge_agent_chain)
        chatbot.add_node("specific_assistance", self.__specific_knowledge_agent_chain)
        chatbot.add_conditional_edges("llm_router", self.__llm_router)
        chatbot.add_edges("general_assistance", END)
        chatbot.add_edges("specific_assistance", END)
        compiled_graph = chatbot.compile_graph(checkpointer=self.__checkpointer)
        g = compiled_graph.get_graph().draw_mermaid_png()
        with open("chatbot_v2.png", "w+b") as f:
            f.write(g)
        self.__chat_states[thread_id] = compiled_graph
        return compiled_graph
    
    def get_chatbot_v2(self, thread_id: str) -> CompiledStateGraph | None:
        try:
            compiled_graph = self.__chat_states.get(thread_id) 
            return compiled_graph
        except Exception:
            raise


