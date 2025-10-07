"""
Main pipeline for SkillGapAI document processing (Milestone 1).
"""

# Import file readers and cleaner modules
from .file_readers_pdf import read_pdf
from .file_readers_docx import read_docx
from .file_readers_txt import read_txt
from .txt_cleaner import clean_text

def process_document(file_bytes, file_type):
	"""Process a single document: extract and clean text."""
	if file_type == 'pdf':
		raw_text = read_pdf(file_bytes)
	elif file_type == 'docx':
		raw_text = read_docx(file_bytes)
	elif file_type == 'txt':
		raw_text = read_txt(file_bytes)
	else:
		raise ValueError(f"Unsupported file type: {file_type}")
	cleaned = clean_text(raw_text)
	return cleaned

def main_pipeline(files):
	"""Run the pipeline on a list of files. Each file is a dict with 'bytes' and 'type'."""
	results = []
	for file in files:
		try:
			cleaned = process_document(file['bytes'], file['type'])
			results.append({'file': file, 'cleaned': cleaned, 'success': True})
		except Exception as e:
			results.append({'file': file, 'error': str(e), 'success': False})
	return results
