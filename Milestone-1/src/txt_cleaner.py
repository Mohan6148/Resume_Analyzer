"""
Text cleaning utilities for SkillGapAI.
"""

import re

def clean_text(text):
    """Clean and preprocess text: remove extra whitespace, normalize line breaks, remove non-printable chars."""
    if not text:
        return ""
    # Remove non-printable characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    # Normalize line breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Remove extra spaces
    text = re.sub(r' +', ' ', text)
    # Remove extra blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leading/trailing whitespace
    return text.strip()
