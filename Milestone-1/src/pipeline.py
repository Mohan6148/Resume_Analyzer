"""
Orchestration pipeline for SkillGapAI (optional, for modularity).
"""

from .complete_pipeline import main_pipeline

def run_pipeline(files):
    """Run the main processing pipeline on uploaded files."""
    return main_pipeline(files)
