from typing import List, Optional
from fastapi import Request, UploadFile


class FileUploadForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.errors: List = []
        self.file: Optional[UploadFile] = None

    async def load_data(self):
        """Load data to form"""
        form = await self.request.form()
        self.file = form.get("file")

    async def file_is_valid(self):
        if self.file is None:
            self.errors.append("File is required")
        if "audio" not in self.file.content_type:
            self.errors.append("The file format is not supported. Please upload an audio file.")
        if not self.errors:
            return True
        return False
