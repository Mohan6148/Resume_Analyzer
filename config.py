"""
Configuration file for SkillGapAI - Resume Skill Analysis System
"""

# Application Configuration
APP_TITLE = "SkillGapAI"
APP_ICON = "🎯"
PAGE_LAYOUT = "wide"

# Model Configuration
SPACY_MODEL = "en_core_web_sm"
SBERT_MODEL = "all-MiniLM-L6-v2"  # Lightweight and fast
SIMILARITY_THRESHOLD = 0.6
CONFIDENCE_THRESHOLD = 0.6

# Skill Categories
SKILL_CATEGORIES = {
    "Programming": ["python", "java", "javascript", "c++", "c#", "go", "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab"],
    "Web Development": ["html", "css", "react", "angular", "vue", "node.js", "django", "flask", "express", "spring", "laravel"],
    "Data Science": ["machine learning", "deep learning", "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn", "keras", "xgboost"],
    "Cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform", "ansible"],
    "Databases": ["sql", "mysql", "postgresql", "mongodb", "redis", "oracle", "dynamodb", "cassandra"],
    "Soft Skills": ["communication", "leadership", "teamwork", "problem solving", "critical thinking", "creativity", "adaptability"]
}

# Allowed File Types
ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# OCR Configuration
USE_OCR = True
OCR_CONFIDENCE_THRESHOLD = 60

# Feature Flags
ENABLE_OCR = True
ENABLE_CUSTOM_NER = True
ENABLE_SBERT = True

