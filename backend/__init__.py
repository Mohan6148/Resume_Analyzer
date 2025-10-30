"""
Backend components for SkillGapAI
"""

from .data_ingestion import DataIngestion
from .skill_extraction import SkillExtractor
from .gap_analysis import SkillGapAnalyzer
from .visualizations import Visualizer
from .export_manager import ExportManager

__all__ = [
    "DataIngestion",
    "SkillExtractor", 
    "SkillGapAnalyzer",
    "Visualizer",
    "ExportManager"
]

