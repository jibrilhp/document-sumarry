from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings


class GenerativeAdapter:
    def __init__(self):
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
