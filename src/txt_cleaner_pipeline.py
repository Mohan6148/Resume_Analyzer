"""
Pipeline for cleaning text files in SkillGapAI.
"""

from txt_cleaner import clean_text

def run_cleaner_pipeline(text):
    """Run the text cleaning pipeline (can add more steps in future)."""
    cleaned = clean_text(text)
    # Add more cleaning steps here if needed
    return cleaned
