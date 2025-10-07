import pdfplumber
from io import BytesIO

def read_pdf(file_bytes):
    """Extract text from PDF file bytes (for Streamlit uploads)."""
    content = ""
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                content += page.extract_text() or ""
        return content.strip()
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return ""