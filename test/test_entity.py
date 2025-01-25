import entity
import unittest

import entity.document

class TestDocumentDb(unittest.TestCase):
    def setUp(self):
        self.document = entity.document.DocumentDb()

    def test_multitenancy_attr_with_str(self):
        self.document.set_multinancy_attr(project_uuid="test_uuid", tenant_id="test_tenant_id")
        self.assertTrue(self.document.project_uuid.__len__() != 0, "document's project uuid is not assigned")
        self.assertTrue(self.document.tenant_id.__len__() != 0, "document's tenant id is not assigned")

class TestChat(unittest.TestCase):
    def setUp(self):
        self.chat = entity.document.Chat(chat="", is_stream=True)
    
    def test_multitenacy_attr_with_str(self):
        self.chat.set_multinancy_attr(project_uuid="test_uuid", tenant_id="test_tenant_id")
        self.assertTrue(self.chat.project_uuid.__len__() != 0, "chat's project uuid is not assigned")
        self.assertTrue(self.chat.tenant_id.__len__() != 0, "chat's tenant id is not assigned")
