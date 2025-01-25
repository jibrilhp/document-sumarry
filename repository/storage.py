from entity.document import Document, DocumentDb
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LangchainDocument
from typing import List
from pathlib import Path

class StorageRepository:
    def __init__(self):
        self.__FILE_PATH__ = "storage"

    def store_document(self, document: Document):
        Path("{}/{}".format(self.__FILE_PATH__, document.project_uuid)).mkdir(parents=True, exist_ok=True)
        document.file.save("{}/{}/{}".format(self.__FILE_PATH__, document.project_uuid, document.file.filename))

    async def load_pdf_document_with_langchain(self, document: DocumentDb)-> List[LangchainDocument]:
        loader = PyPDFLoader("{}/{}/{}".format(self.__FILE_PATH__, document.project_uuid, document.document_name))
        pages: List[LangchainDocument] = list()
        async for page in loader.alazy_load():
            page.metadata["tenant_id"] = document.tenant_id
            page.metadata["project_uuid"] = document.project_uuid
            print("metadata", page.metadata)
            pages.append(page)
        return pages
