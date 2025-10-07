"""
Task 3: Skill Extraction
Milestone 2: NLP & Skill Extraction Tasks

This module contains functions for skill extraction operations:
1. Create Skill Database
2. Simple Skill Matcher
3. Handle Skill Abbreviations

Author: Mohan Sri Sai Panduri
ID: 22A81A6148@sves.org.in
"""

import re
from typing import Dict, List


# Question 3.1: Create Skill Database (8 points)
SKILL_DATABASE = {
    'programming_languages': [
        'Python', 'Java', 'JavaScript', 'C++', 'C#', 'Ruby', 'PHP', 'Go', 'Rust', 'Swift',
        'Kotlin', 'Scala', 'TypeScript', 'C', 'R', 'MATLAB', 'Perl', 'Shell', 'Bash', 'PowerShell'
    ],
    'frameworks': [
        'TensorFlow', 'PyTorch', 'React', 'Angular', 'Vue.js', 'Django', 'Flask', 'Spring',
        'Laravel', 'Express.js', 'Node.js', 'ASP.NET', 'Ruby on Rails', 'FastAPI', 'Keras',
        'Scikit-learn', 'Pandas', 'NumPy', 'Bootstrap', 'jQuery'
    ],
    'databases': [
        'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Oracle', 'SQLite', 'Cassandra', 'Elasticsearch',
        'Neo4j', 'DynamoDB', 'MariaDB', 'SQL Server', 'Firebase', 'CouchDB', 'InfluxDB'
    ],
    'cloud': [
        'AWS', 'Azure', 'Google Cloud Platform', 'IBM Cloud', 'Oracle Cloud', 'DigitalOcean',
        'Heroku', 'Vercel', 'Netlify', 'Firebase', 'Cloudflare', 'Alibaba Cloud', 'Linode'
    ],
    'soft_skills': [
        'Leadership', 'Communication', 'Problem Solving', 'Teamwork', 'Time Management',
        'Adaptability', 'Creativity', 'Critical Thinking', 'Project Management', 'Analytical Skills',
        'Attention to Detail', 'Customer Service', 'Negotiation', 'Mentoring', 'Presentation Skills'
    ]
}

# Question 3.3: Handle Skill Abbreviations (9 points)
ABBREVIATIONS = {
    'ML': 'Machine Learning',
    'DL': 'Deep Learning', 
    'NLP': 'Natural Language Processing',
    'AI': 'Artificial Intelligence',
    'JS': 'JavaScript',
    'K8s': 'Kubernetes',
    'GCP': 'Google Cloud Platform',
    'API': 'Application Programming Interface',
    'SQL': 'Structured Query Language',
    'NoSQL': 'Not Only SQL',
    'REST': 'Representational State Transfer',
    'CRUD': 'Create, Read, Update, Delete',
    'CI/CD': 'Continuous Integration/Continuous Deployment',
    'DevOps': 'Development Operations',
    'UI': 'User Interface',
    'UX': 'User Experience',
    'VR': 'Virtual Reality',
    'AR': 'Augmented Reality',
    'IoT': 'Internet of Things',
    'DBA': 'Database Administrator'
}


def extract_skills(text, skill_database):
    """
    Question 3.2: Simple Skill Matcher (8 points)
    
    Searches for skills from the database in the text and categorizes them.
    
    Args:
        text (str): Input text to search
        skill_database (dict): Dictionary containing skill categories
        
    Returns:
        dict: Found skills categorized by type
    """
    if not text:
        return {}
    
    found_skills = {
        'programming_languages': [],
        'frameworks': [],
        'databases': [],
        'cloud': [],
        'soft_skills': []
    }
    
    text_lower = text.lower()
    
    # Search through each category
    for category, skills in skill_database.items():
        for skill in skills:
            # Case-insensitive matching
            if skill.lower() in text_lower:
                found_skills[category].append(skill)
    
    # Remove empty categories
    found_skills = {k: v for k, v in found_skills.items() if v}
    
    return found_skills


def normalize_skills(skill_list):
    """
    Question 3.3: Handle Skill Abbreviations (9 points)
    
    Converts abbreviations to full names using the abbreviations mapping.
    
    Args:
        skill_list (list): List of skills that may contain abbreviations
        
    Returns:
        list: List of skills with abbreviations expanded
    """
    if not skill_list:
        return []
    
    normalized_skills = []
    
    for skill in skill_list:
        # Check if skill is an abbreviation
        if skill.upper() in ABBREVIATIONS:
            normalized_skills.append(ABBREVIATIONS[skill.upper()])
        else:
            # Keep original skill if not an abbreviation
            normalized_skills.append(skill)
    
    return normalized_skills


def get_all_skills_from_database():
    """
    Helper function to get all skills from the database as a flat list.
    
    Returns:
        list: All skills from the database
    """
    all_skills = []
    for category_skills in SKILL_DATABASE.values():
        all_skills.extend(category_skills)
    return all_skills


def extract_skills_with_normalization(text, skill_database):
    """
    Enhanced skill extraction that also normalizes abbreviations.
    
    Args:
        text (str): Input text to search
        skill_database (dict): Dictionary containing skill categories
        
    Returns:
        dict: Found skills with abbreviations normalized
    """
    # First extract skills normally
    found_skills = extract_skills(text, skill_database)
    
    # Normalize abbreviations in each category
    for category, skills in found_skills.items():
        found_skills[category] = normalize_skills(skills)
    
    return found_skills


def test_functions():
    """
    Test function to verify all skill extraction functions work correctly.
    """
    print("=== Testing Task 3: Skill Extraction Functions ===\n")
    
    # Test 3.1: Skill Database
    print("3.1 Testing SKILL_DATABASE:")
    print(f"Programming Languages: {len(SKILL_DATABASE['programming_languages'])} skills")
    print(f"Frameworks: {len(SKILL_DATABASE['frameworks'])} skills")
    print(f"Databases: {len(SKILL_DATABASE['databases'])} skills")
    print(f"Cloud: {len(SKILL_DATABASE['cloud'])} skills")
    print(f"Soft Skills: {len(SKILL_DATABASE['soft_skills'])} skills")
    print("✓ Database created successfully\n")
    
    # Test 3.2: Simple Skill Matcher
    print("3.2 Testing extract_skills():")
    test_text = "Proficient in Python, Java, TensorFlow, and AWS. Strong leadership skills."
    result = extract_skills(test_text, SKILL_DATABASE)
    print(f"Input: {test_text}")
    print(f"Output: {result}")
    expected = {
        'programming_languages': ['Python', 'Java'],
        'frameworks': ['TensorFlow'],
        'cloud': ['AWS'],
        'soft_skills': ['Leadership']
    }
    print(f"Expected: {expected}")
    print(f"Match: {'✓' if result == expected else '✗'}\n")
    
    # Test 3.3: Skill Abbreviations
    print("3.3 Testing normalize_skills():")
    test_skills = ['ML', 'DL', 'NLP', 'JS', 'K8s', 'AWS', 'GCP']
    result = normalize_skills(test_skills)
    print(f"Input: {test_skills}")
    print(f"Output: {result}")
    expected = ['Machine Learning', 'Deep Learning', 'Natural Language Processing', 
               'JavaScript', 'Kubernetes', 'AWS', 'Google Cloud Platform']
    print(f"Expected: {expected}")
    print(f"Match: {'✓' if result == expected else '✗'}\n")


def demonstrate_skill_extraction():
    """
    Additional demonstration of skill extraction capabilities.
    """
    print("=== Skill Extraction Demonstration ===\n")
    
    sample_resume_text = """
    I am a Senior Software Engineer with expertise in Python, Java, and JavaScript.
    I have worked extensively with Machine Learning frameworks like TensorFlow and PyTorch.
    I am proficient in databases including MySQL, MongoDB, and Redis.
    I have cloud experience with AWS, Azure, and GCP.
    My soft skills include leadership, communication, and problem-solving abilities.
    I also have experience with CI/CD, REST APIs, and DevOps practices.
    """
    
    print("Sample Resume Text:")
    print(sample_resume_text)
    
    # Extract skills
    print("Extracted Skills:")
    found_skills = extract_skills_with_normalization(sample_resume_text, SKILL_DATABASE)
    for category, skills in found_skills.items():
        if skills:
            print(f"  {category.replace('_', ' ').title()}: {', '.join(skills)}")
    
    print(f"\nTotal Skills Found: {sum(len(skills) for skills in found_skills.values())}")


def show_abbreviation_mappings():
    """
    Display all abbreviation mappings.
    """
    print("=== Skill Abbreviation Mappings ===\n")
    for abbrev, full_name in ABBREVIATIONS.items():
        print(f"  {abbrev} → {full_name}")


if __name__ == "__main__":
    test_functions()
    print("\n" + "="*50 + "\n")
    demonstrate_skill_extraction()
    print("\n" + "="*50 + "\n")
    show_abbreviation_mappings()
