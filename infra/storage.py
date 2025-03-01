import re
from entity.document import Document, DocumentDb
from pypdf import PdfReader
from langchain_core.documents import Document as LangchainDocument
from typing import List
from pathlib import Path
from PIL.Image import open as open_image
from pytesseract.pytesseract import image_to_string
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

class StorageRepository:
    def __init__(self):
        self.__FILE_PATH__ = "storage"

    def store_document(self, document: Document):
        Path("{}/{}".format(self.__FILE_PATH__, document.project_uuid)).mkdir(parents=True, exist_ok=True)
        document.file.file.seek(0)
        file_loc = "{}/{}/{}".format(self.__FILE_PATH__, document.project_uuid, document.file.filename)
        with open(file_loc, "wb") as out:
            out.write(document.file.file.read())

    async def load_pdf_document_with_langchain(self, document: DocumentDb)-> List[LangchainDocument]:
        file_path = "{}/{}/{}".format(self.__FILE_PATH__, document.project_uuid, document.document_name)
        logging.info("load pdf from {}".format(file_path))
        reader = PdfReader(file_path)
        extracted_text: str = ""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500_000, chunk_overlap=10_000)
        metadatas: List[dict] = list()
        for page in reader.pages:
            extracted_text += self.clean_text(page.extract_text())
            metadatas.append({"tenant_id": document.tenant_id, "project_uuid": document.project_uuid})
        splitted_texts = text_splitter.split_text(extracted_text)
        documents = text_splitter.create_documents(splitted_texts, metadatas)
        return documents
    
    def clean_text(self, text: str):
        text = re.sub(r'\s+', ' ', text).strip()
        return text


    def load_image_with_langchain(self, image_db: DocumentDb) -> List[LangchainDocument]:
        image = open_image("{}/{}/{}".format(self.__FILE_PATH__, image_db.project_uuid, image_db.document_name))
        text_from_image: str = image_to_string(image=image)
        langchain_images: List[LangchainDocument] = list()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        page_chunks = text_splitter.split_text(text_from_image)
        for chunk in page_chunks:
            langchain_image: LangchainDocument = LangchainDocument(page_content=chunk)
            langchain_image.metadata["tenant_id"] = image_db.tenant_id
            langchain_image.metadata["project_uuid"] = image_db.project_uuid
            langchain_images.append(langchain_image)
        return langchain_images