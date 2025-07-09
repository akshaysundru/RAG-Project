from docx import Document
import os
from datetime import datetime
from docx.enum.text import WD_ALIGN_PARAGRAPH

class DocumentLogger:
    def __init__(self, session_id, output_dir = "./chat_logs"):
        self.session_id = session_id
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.doc_path = os.path.join(output_dir, f"chat_session_{session_id}.docx")

        if os.path.exists(self.doc_path):
            self.document = Document(self.doc_path)
        else:
            self.document = Document()
            self.document.add_heading(f"Chat_Session_{session_id}", 0)
            self.document.save(self.doc_path)

    def write_interaction(self, question, response, sources = None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.document.add_heading(f'User Question ({timestamp}):', level=2)

        self.document.add_paragraph(question)
        
        self.document.add_heading('AI Response:', level=2)
        self.document.add_paragraph(response)
        
        if sources:
            self.document.add_heading('Sources:', level=2)
            self.document.add_paragraph(sources)
        
        self.document.add_page_break()
        self.document.save(self.doc_path)

class DocumentWriter:
    def __init__(self, filename, output_dir = "./chat_logs"):
        self.filename = filename
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.doc_path = os.path.join(output_dir, f"{filename}.docx")

        if os.path.exists(self.doc_path):
            self.document = Document(self.doc_path)
        else:
            self.document = Document()
            title = self.document.add_heading(f"{filename}", 0)
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            self.document.save(self.doc_path)

    def add_heading(self, user_input):

        heading = self.document.add_heading(user_input, 1)
        heading.alignment = WD_ALIGN_PARAGRAPH.LEFT

    def add_subheading(self, user_input):

        heading = self.document.add_heading(user_input, 2)
        heading.alignment = WD_ALIGN_PARAGRAPH.LEFT