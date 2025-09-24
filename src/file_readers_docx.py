import docx
from io import BytesIO

def read_docx(file_bytes):
    """Extract text from DOCX file bytes (for Streamlit uploads)."""
    try:
        doc = docx.Document(BytesIO(file_bytes))
        content = "\n".join([p.text for p in doc.paragraphs])
        return content
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
        return ""