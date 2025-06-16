class UnknownFileType(Exception):
    def __init__(self, message: str = "invalid file type provided", file_type: str = ""):
        super().__init__(message)
        self.file_type = file_type


class FileConflictDb(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class DatabaseError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class ResourceNotFound(Exception):
    def __init__(self, message):
        super().__init__(message)


class FileTooLarge(Exception):
    def __init__(self, message):
        super().__init__(message)

class UnauthorizedAccess(Exception):
    def __init__(self, message):
        super().__init__(message)