"""
Data Ingestion & Parsing
Handles file uploads, text extraction, cleaning, and normalization
"""

try:
    import streamlit as st
except Exception:
    # Minimal fallback when running outside Streamlit / for editor checks
    class _StFallback:
        def error(self, msg):
            print("[STREAMLIT ERROR]", msg)
        def warning(self, msg):
            print("[STREAMLIT WARNING]", msg)
        def info(self, msg):
            print("[STREAMLIT INFO]", msg)
        def success(self, msg):
            print("[STREAMLIT SUCCESS]", msg)

    st = _StFallback()

try:
    import pandas as pd
except Exception:
    pd = None
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import io
import re

# PDF Processing
try:
    import PyPDF2
    import pdfplumber
except ImportError:
    st.error("PyPDF2 and pdfplumber not installed. Run: pip install PyPDF2 pdfplumber")

# DOCX Processing  
try:
    from docx import Document
except ImportError:
    st.error("python-docx not installed. Run: pip install python-docx")

# OCR for Scanned PDFs
try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_bytes
except ImportError:
    # OCR dependencies not installed (optional)
    pytesseract = None
    Image = None
    convert_from_bytes = None


class DataIngestion:
    """Handles document parsing, cleaning, and text extraction"""
    
    def __init__(self):
        self.uploaded_files = {}
        self.parsed_documents = {}
        self.errors = []
        
    def upload_files(self, uploaded_files: List) -> Dict:
        """Track uploaded files with metadata"""
        for file in uploaded_files:
            if file not in st.session_state.get('uploaded_files', []):
                file_info = {
                    'name': file.name,
                    'type': Path(file.name).suffix,
                    'size': len(file.getvalue()),
                    'status': 'pending',
                    'pages': 0,
                    'word_count': 0,
                    'raw_text': None,
                    'cleaned_text': None,
                    'error': None
                }
                self.uploaded_files[file.name] = file_info
        
        return self.uploaded_files
    
    def parse_document(self, file, file_info: Dict) -> Tuple[str, str, Optional[str]]:
        """Parse document based on file type and return raw, cleaned text, and error"""
        try:
            file_ext = Path(file.name).suffix.lower()
            
            if file_ext == '.pdf':
                raw_text, error = self._parse_pdf(file)
            elif file_ext == '.docx':
                raw_text, error = self._parse_docx(file)
            elif file_ext == '.txt':
                raw_text, error = self._parse_txt(file)
            else:
                return "", "", "Unsupported file type"
            
            if error:
                return "", "", error
                
            cleaned_text = self._clean_text(raw_text)
            word_count = len(cleaned_text.split())
            
            file_info['raw_text'] = raw_text
            file_info['cleaned_text'] = cleaned_text
            file_info['word_count'] = word_count
            file_info['status'] = 'success'
            
            return raw_text, cleaned_text, None
            
        except Exception as e:
            error_msg = f"Parse error: {str(e)}"
            file_info['error'] = error_msg
            file_info['status'] = 'error'
            return "", "", error_msg
    
    def _parse_pdf(self, file) -> Tuple[str, Optional[str]]:
        """Extract text from PDF, attempt OCR if needed"""
        try:
            # Try text-based extraction first
            raw_text = ""
            pdf_file = io.BytesIO(file.getvalue())
            
            # Try pdfplumber (better text extraction)
            try:
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            raw_text += text + "\n\n"
                
                if len(raw_text.strip()) > 100:
                    return raw_text, None
            except:
                pass
            
            # Fallback to PyPDF2
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text()
                except Exception:
                    page_text = None
                if page_text:
                    raw_text += page_text + "\n\n"
            
            # If extracted text is minimal, try OCR
            if len(raw_text.strip()) < 100:
                ocr_text = self._extract_with_ocr(file)
                if ocr_text:
                    return ocr_text, None
            
            return raw_text, None
            
        except Exception as e:
            return "", f"PDF parsing error: {str(e)}"
    
    def _parse_docx(self, file) -> Tuple[str, Optional[str]]:
        """Extract text from DOCX file"""
        try:
            doc = Document(io.BytesIO(file.getvalue()))
            raw_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return raw_text, None
        except Exception as e:
            return "", f"DOCX parsing error: {str(e)}"
    
    def _parse_txt(self, file) -> Tuple[str, Optional[str]]:
        """Extract text from TXT file"""
        try:
            raw_text = file.read().decode('utf-8')
            return raw_text, None
        except Exception as e:
            return "", f"TXT parsing error: {str(e)}"
    
    def _extract_with_ocr(self, file) -> Optional[str]:
        """Use OCR to extract text from scanned PDF"""
        try:
            if pytesseract is None or convert_from_bytes is None:
                return None
            
            images = convert_from_bytes(file.getvalue())
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img) + "\n"
            
            return ocr_text if len(ocr_text.strip()) > 50 else None
        except Exception:
            # OCR dependencies not installed or error occurred
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep alphanumeric, punctuation, and newlines
        text = re.sub(r'[^\w\s.,;:!?\-\n\()]', ' ', text)
        
        # Remove headers/footers (lines with only numbers or dashes)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line_stripped = line.strip()
            # Skip lines that are just numbers, dashes, or very short
            if not re.match(r'^[\d\-_\s]+$', line_stripped) and len(line_stripped) > 2:
                cleaned_lines.append(line_stripped)
        
        return '\n'.join(cleaned_lines)
    
    def get_normalization_summary(self, file_info: Dict) -> Dict:
        """Get summary stats about normalization"""
        raw_text = file_info.get('raw_text', '')
        cleaned_text = file_info.get('cleaned_text', '')
        
        original_lines = len(raw_text.split('\n'))
        cleaned_lines = len(cleaned_text.split('\n'))
        
        return {
            'lines_removed': max(0, original_lines - cleaned_lines),
            'original_length': len(raw_text),
            'cleaned_length': len(cleaned_text),
            'word_count': file_info.get('word_count', 0)
        }

