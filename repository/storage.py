from entity.document import Document

class StorageRepository:
    def __init__(self):
        self.__FILE_PATH__ = "storage"

    def store_document(self, document: Document):
        document.file.save("{}/{}".format(self.__FILE_PATH__, document.file.filename))