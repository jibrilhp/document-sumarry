from entity.document import Document, DocumentDb
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LangchainDocument
from typing import List
from pathlib import Path
from PIL.Image import open as open_image
from pytesseract.pytesseract import image_to_string
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil

class StorageRepository:
    def __init__(self):
        self.__FILE_PATH__ = "storage"

    def store_document(self, document: Document):
        Path("{}/{}".format(self.__FILE_PATH__, document.project_uuid)).mkdir(parents=True, exist_ok=True)
        with open("{}/{}/{}".format(self.__FILE_PATH__, document.project_uuid, document.file.filename), "wb") as buffer:
            shutil.copyfileobj(document.file.file, buffer)  # Efficient file streaming

    async def load_pdf_document_with_langchain(self, document: DocumentDb)-> List[LangchainDocument]:
        loader = PyPDFLoader("{}/{}/{}".format(self.__FILE_PATH__, document.project_uuid, document.document_name))
        pages: List[LangchainDocument] = list()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        iter_document = loader.lazy_load()
        page_chunks = text_splitter.split_documents(iter_document)
        for page_chunk in page_chunks:
            page_chunk.metadata["tenant_id"] = document.tenant_id
            page_chunk.metadata["project_uuid"] = document.project_uuid
            pages.append(page_chunk)
        return pages
    
    def load_image_with_langchain(self, image_db: DocumentDb) -> List[LangchainDocument]:
        image = open_image("{}/{}/{}".format(self.__FILE_PATH__, image_db.project_uuid, image_db.document_name))
        text_from_image: str = image_to_string(image=image)
        langchain_images: List[LangchainDocument] = list()
        langchain_image: LangchainDocument = LangchainDocument(page_content=text_from_image)
        langchain_image.metadata["tenant_id"] = image_db.tenant_id
        langchain_image.metadata["project_uuid"] = image_db.project_uuid
        langchain_images.append(langchain_image)
        return langchain_images