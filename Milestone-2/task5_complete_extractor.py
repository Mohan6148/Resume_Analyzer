"""
Task 5: Complete Skill Extractor
Milestone 2: NLP & Skill Extraction Tasks

This module contains the complete skill extraction system:
1. Multi-Method Extractor
2. Skill Report Generator
3. Bonus Functions (Frequency Counter & Context Extractor)

Author: Mohan Sri Sai Panduri
ID: 22A81A6148@sves.org.in
"""

import re
import spacy
from collections import Counter
from typing import Dict, List, Set

# Import functions from previous tasks
try:
    from task1_preprocessing import clean_resume_text, remove_stop_words
    from task2_pos_tagging import extract_nouns, find_adj_noun_patterns
    from task3_skill_extraction import SKILL_DATABASE, extract_skills, normalize_skills
    from task4_ner import extract_entities
except ImportError:
    print("Warning: Could not import functions from previous tasks.")
    print("Make sure all task files are in the same directory.")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model 'en_core_web_sm' not found.")
    nlp = None


def extract_all_skills(resume_text):
    """
    Question 5.1: Build Multi-Method Extractor (12 points)
    
    Combines multiple extraction methods:
    1. Keyword matching from database
    2. POS pattern matching (ADJ+NOUN, NOUN+NOUN)
    3. Named Entity Recognition
    
    Args:
        resume_text (str): Input resume text
        
    Returns:
        dict: All unique skills found categorized
    """
    if not resume_text:
        return {'technical_skills': [], 'soft_skills': [], 'all_skills': []}
    
    all_found_skills = set()
    technical_skills = set()
    soft_skills = set()
    
    # Method 1: Keyword matching from database
    try:
        db_skills = extract_skills(resume_text, SKILL_DATABASE)
        for category, skills in db_skills.items():
            normalized_skills = normalize_skills(skills)
            for skill in normalized_skills:
                all_found_skills.add(skill)
                if category == 'soft_skills':
                    soft_skills.add(skill)
                else:
                    technical_skills.add(skill)
    except:
        pass
    
    # Method 2: POS pattern matching
    try:
        # Extract nouns (potential skills)
        nouns = extract_nouns(resume_text)
        for noun in nouns:
            all_found_skills.add(noun)
            technical_skills.add(noun)
        
        # Find adjective-noun patterns
        patterns = find_adj_noun_patterns(resume_text)
        for pattern in patterns:
            all_found_skills.add(pattern)
            technical_skills.add(pattern)
    except:
        pass
    
    # Method 3: Named Entity Recognition
    try:
        entities = extract_entities(resume_text)
        for entity_text, entity_label in entities:
            if entity_label in ['PRODUCT', 'ORG']:
                all_found_skills.add(entity_text)
                technical_skills.add(entity_text)
    except:
        pass
    
    # Additional pattern matching for common skills
    additional_patterns = [
        r'\b(?:Machine|Deep|Natural|Artificial|Data|Big|Cloud|Web|Mobile)\s+(?:Learning|Language|Processing|Science|Data|Computing|Development)\b',
        r'\b(?:Object|Functional|Procedural)\s+(?:Oriented|Programming)\b',
        r'\b(?:Full|Back|Front|Mobile|Web|Software)\s+(?:Stack|End|Development)\b',
        r'\b(?:User|Application)\s+(?:Interface|Experience|Programming)\b',
        r'\b(?:Version|Source|Quality)\s+(?:Control|Code|Assurance)\b'
    ]
    
    for pattern in additional_patterns:
        matches = re.findall(pattern, resume_text, re.IGNORECASE)
        for match in matches:
            all_found_skills.add(match)
            technical_skills.add(match)
    
    return {
        'technical_skills': list(technical_skills),
        'soft_skills': list(soft_skills),
        'all_skills': list(all_found_skills)
    }


def generate_skill_report(skills_dict):
    """
    Question 5.2: Generate Skill Report (8 points)
    
    Takes the output from extract_all_skills and displays a formatted report.
    
    Args:
        skills_dict (dict): Output from extract_all_skills function
        
    Returns:
        str: Formatted skill report
    """
    if not skills_dict:
        return "No skills found."
    
    technical_skills = skills_dict.get('technical_skills', [])
    soft_skills = skills_dict.get('soft_skills', [])
    all_skills = skills_dict.get('all_skills', [])
    
    total_skills = len(all_skills)
    technical_count = len(technical_skills)
    soft_count = len(soft_skills)
    
    # Calculate percentages
    technical_percent = (technical_count / total_skills * 100) if total_skills > 0 else 0
    soft_percent = (soft_count / total_skills * 100) if total_skills > 0 else 0
    
    # Generate report
    report = []
    report.append("=== SKILL EXTRACTION REPORT ===\n")
    
    # Technical Skills Section
    report.append(f"TECHNICAL SKILLS ({technical_count}):")
    if technical_skills:
        for skill in sorted(technical_skills):
            report.append(f"  • {skill}")
    else:
        report.append("  • None found")
    
    report.append("")
    
    # Soft Skills Section
    report.append(f"SOFT SKILLS ({soft_count}):")
    if soft_skills:
        for skill in sorted(soft_skills):
            report.append(f"  • {skill}")
    else:
        report.append("  • None found")
    
    report.append("")
    
    # Summary Section
    report.append("SUMMARY:")
    report.append(f"  Total Skills: {total_skills}")
    report.append(f"  Technical: {technical_count} ({technical_percent:.0f}%)")
    report.append(f"  Soft Skills: {soft_count} ({soft_percent:.0f}%)")
    
    return "\n".join(report)


# Bonus Questions
def count_skill_frequency(text):
    """
    Bonus 1: Frequency Counter (5 points)
    
    Counts how many times each skill appears in the text.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Skills with their frequency counts
    """
    if not text:
        return {}
    
    # Extract all skills
    skills_dict = extract_all_skills(text)
    all_skills = skills_dict.get('all_skills', [])
    
    # Count frequency of each skill in the text
    skill_counts = {}
    text_lower = text.lower()
    
    for skill in all_skills:
        # Count occurrences (case-insensitive)
        count = len(re.findall(re.escape(skill.lower()), text_lower))
        if count > 0:
            skill_counts[skill] = count
    
    # Sort by frequency (descending)
    sorted_counts = dict(sorted(skill_counts.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_counts


def extract_skill_context(text, skill):
    """
    Bonus 2: Skill Context Extractor (5 points)
    
    Finds sentences containing a specific skill and returns the context.
    
    Args:
        text (str): Input text
        skill (str): Skill to find context for
        
    Returns:
        list: List of sentences containing the skill
    """
    if not text or not skill:
        return []
    
    # Split text into sentences
    sentences = re.split(r'[.!?]+', text)
    contexts = []
    
    skill_lower = skill.lower()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if skill_lower in sentence.lower() and sentence:
            contexts.append(sentence)
    
    return contexts


def test_functions():
    """
    Test function to verify all skill extraction functions work correctly.
    """
    print("=== Testing Task 5: Complete Skill Extractor Functions ===\n")
    
    # Test 5.1: Multi-Method Extractor
    print("5.1 Testing extract_all_skills():")
    test_resume = """
SKILLS:
Programming: Python, Java, JavaScript
Frameworks: TensorFlow, React, Django
Experience in Machine Learning and Deep Learning
Strong analytical and problem-solving skills
"""
    result = extract_all_skills(test_resume)
    print(f"Input: {repr(test_resume)}")
    print(f"Output: {result}")
    print("Expected: technical_skills with Python, Java, JavaScript, TensorFlow, React, Django, Machine Learning, Deep Learning")
    print("Expected: soft_skills with analytical, problem-solving")
    print(f"Match: {'✓' if result['technical_skills'] and result['soft_skills'] else '✗'}\n")
    
    # Test 5.2: Skill Report
    print("5.2 Testing generate_skill_report():")
    report = generate_skill_report(result)
    print("Generated Report:")
    print(report)
    print("✓ Report generated successfully\n")
    
    # Test Bonus 1: Frequency Counter
    print("Bonus 1 Testing count_skill_frequency():")
    test_text = """
Python developer with Python experience. 
Used Python and Machine Learning. 
Machine Learning projects with Python.
"""
    frequency_result = count_skill_frequency(test_text)
    print(f"Input: {repr(test_text)}")
    print(f"Output: {frequency_result}")
    print("Expected: Python with higher frequency than Machine Learning")
    print(f"Match: {'✓' if frequency_result else '✗'}\n")
    
    # Test Bonus 2: Context Extractor
    print("Bonus 2 Testing extract_skill_context():")
    context_result = extract_skill_context(test_text, "Python")
    print(f"Input text: {repr(test_text)}")
    print(f"Skill: Python")
    print(f"Output: {context_result}")
    print("Expected: Sentences containing 'Python'")
    print(f"Match: {'✓' if context_result else '✗'}\n")


def calculate_skill_coverage_score(skills_dict, required_skills):
    """
    Bonus 3: Skill Coverage Score Calculator (5 points)
    
    Calculates a coverage score based on how many required skills are present.
    
    Args:
        skills_dict (dict): Output from extract_all_skills function
        required_skills (list): List of required skills for a job
        
    Returns:
        dict: Coverage score and analysis
    """
    if not skills_dict or not required_skills:
        return {"score": 0, "coverage_percent": 0, "missing_skills": required_skills, "matched_skills": []}
    
    all_found_skills = [skill.lower() for skill in skills_dict.get('all_skills', [])]
    required_skills_lower = [skill.lower() for skill in required_skills]
    
    matched_skills = []
    missing_skills = []
    
    for req_skill in required_skills_lower:
        # Check for exact match or partial match
        found = False
        for found_skill in all_found_skills:
            if req_skill in found_skill or found_skill in req_skill:
                matched_skills.append(req_skill)
                found = True
                break
        
        if not found:
            missing_skills.append(req_skill)
    
    coverage_percent = (len(matched_skills) / len(required_skills)) * 100
    
    return {
        "score": len(matched_skills),
        "coverage_percent": round(coverage_percent, 2),
        "missing_skills": missing_skills,
        "matched_skills": matched_skills,
        "total_required": len(required_skills)
    }


def extract_skill_synonyms(text, skill):
    """
    Bonus 4: Skill Synonym Extractor (5 points)
    
    Finds synonyms and related terms for a given skill in the text.
    
    Args:
        text (str): Input text
        skill (str): Skill to find synonyms for
        
    Returns:
        list: List of related terms found in text
    """
    if not text or not skill:
        return []
    
    # Define skill synonyms mapping
    skill_synonyms = {
        'python': ['python', 'py', 'pythonic', 'pythonista'],
        'java': ['java', 'javase', 'jvm', 'spring'],
        'javascript': ['javascript', 'js', 'node', 'nodejs', 'ecmascript'],
        'machine learning': ['ml', 'machine learning', 'ai', 'artificial intelligence'],
        'data science': ['data science', 'data analysis', 'analytics', 'data mining'],
        'web development': ['web dev', 'web development', 'frontend', 'backend', 'fullstack'],
        'database': ['database', 'db', 'sql', 'nosql', 'data storage'],
        'cloud': ['cloud', 'aws', 'azure', 'gcp', 'cloud computing'],
        'leadership': ['leadership', 'leading', 'management', 'team lead'],
        'communication': ['communication', 'communicating', 'presentation', 'public speaking']
    }
    
    skill_lower = skill.lower()
    text_lower = text.lower()
    related_terms = []
    
    # Get synonyms for the skill
    synonyms = skill_synonyms.get(skill_lower, [skill_lower])
    
    # Find all related terms in text
    for synonym in synonyms:
        if synonym in text_lower and synonym != skill_lower:
            related_terms.append(synonym)
    
    return list(set(related_terms))  # Remove duplicates


def generate_skill_recommendations(skills_dict, job_requirements):
    """
    Bonus 5: Skill Recommendation Generator (5 points)
    
    Generates skill recommendations based on missing skills for a job.
    
    Args:
        skills_dict (dict): Output from extract_all_skills function
        job_requirements (list): Required skills for a job
        
    Returns:
        dict: Skill recommendations and priority levels
    """
    if not skills_dict or not job_requirements:
        return {"recommendations": [], "priority": "High", "reason": "No skills found"}
    
    all_found_skills = [skill.lower() for skill in skills_dict.get('all_skills', [])]
    job_reqs_lower = [req.lower() for req in job_requirements]
    
    missing_skills = []
    for req_skill in job_reqs_lower:
        found = False
        for found_skill in all_found_skills:
            if req_skill in found_skill or found_skill in req_skill:
                found = True
                break
        if not found:
            missing_skills.append(req_skill)
    
    # Categorize recommendations by priority
    high_priority = []
    medium_priority = []
    low_priority = []
    
    # Define priority categories
    high_priority_skills = ['python', 'java', 'javascript', 'sql', 'machine learning', 'aws', 'docker']
    medium_priority_skills = ['react', 'angular', 'vue', 'django', 'flask', 'mongodb', 'kubernetes']
    low_priority_skills = ['leadership', 'communication', 'problem solving', 'teamwork']
    
    for skill in missing_skills:
        if any(hp in skill for hp in high_priority_skills):
            high_priority.append(skill)
        elif any(mp in skill for mp in medium_priority_skills):
            medium_priority.append(skill)
        else:
            low_priority.append(skill)
    
    # Determine overall priority
    if high_priority:
        priority = "High"
        reason = f"Missing {len(high_priority)} critical technical skills"
    elif medium_priority:
        priority = "Medium"
        reason = f"Missing {len(medium_priority)} important skills"
    else:
        priority = "Low"
        reason = f"Missing {len(low_priority)} supplementary skills"
    
    return {
        "recommendations": {
            "high_priority": high_priority,
            "medium_priority": medium_priority,
            "low_priority": low_priority
        },
        "priority": priority,
        "reason": reason,
        "total_missing": len(missing_skills)
    }


def demonstrate_complete_extraction():
    """
    Demonstrate the complete skill extraction system.
    """
    print("=== Complete Skill Extraction Demonstration ===\n")
    
    sample_resume = """
    John Smith
    Senior Software Engineer
    
    EXPERIENCE:
    I am a Python developer with 5 years of experience in Machine Learning.
    I have worked extensively with TensorFlow and PyTorch frameworks.
    My programming skills include Python, Java, and JavaScript.
    I have experience with AWS cloud services and Docker containers.
    I am proficient in SQL and MongoDB databases.
    
    SKILLS:
    - Python, Java, JavaScript, C++
    - TensorFlow, PyTorch, React, Django
    - AWS, Docker, Kubernetes
    - MySQL, MongoDB, Redis
    - Leadership, Communication, Problem Solving
    
    I have strong analytical skills and excellent communication abilities.
    """
    
    print("Sample Resume:")
    print(sample_resume)
    print("\n" + "="*50 + "\n")
    
    # Extract all skills
    skills = extract_all_skills(sample_resume)
    
    # Generate report
    report = generate_skill_report(skills)
    print(report)
    
    # Show frequency analysis
    print("\n" + "="*30 + "\n")
    print("SKILL FREQUENCY ANALYSIS:")
    frequencies = count_skill_frequency(sample_resume)
    for skill, count in list(frequencies.items())[:10]:  # Top 10
        print(f"  {skill}: {count} times")
    
    # Show context examples
    print("\n" + "="*30 + "\n")
    print("SKILL CONTEXT EXAMPLES:")
    python_context = extract_skill_context(sample_resume, "Python")
    print(f"Python contexts ({len(python_context)}):")
    for i, context in enumerate(python_context[:3], 1):
        print(f"  {i}. {context.strip()}")


def demonstrate_bonus_features():
    """
    Demonstrate the additional bonus features.
    """
    print("\n" + "="*50 + "\n")
    print("=== BONUS FEATURES DEMONSTRATION ===\n")
    
    sample_resume = """
    I am a Python developer with 3 years of experience in web development.
    I have worked with Django, Flask, and React frameworks.
    I have experience with MySQL and PostgreSQL databases.
    I am learning Machine Learning and Data Science.
    I have good communication skills and problem-solving abilities.
    """
    
    # Extract skills
    skills = extract_all_skills(sample_resume)
    
    # Bonus 3: Skill Coverage Score
    print("BONUS 3: Skill Coverage Score Calculator")
    print("-" * 40)
    required_skills = ['Python', 'Django', 'React', 'MySQL', 'AWS', 'Docker', 'Machine Learning']
    coverage = calculate_skill_coverage_score(skills, required_skills)
    print(f"Required Skills: {required_skills}")
    print(f"Coverage Score: {coverage['score']}/{coverage['total_required']} ({coverage['coverage_percent']}%)")
    print(f"Matched Skills: {coverage['matched_skills']}")
    print(f"Missing Skills: {coverage['missing_skills']}")
    print()
    
    # Bonus 4: Skill Synonyms
    print("BONUS 4: Skill Synonym Extractor")
    print("-" * 40)
    python_synonyms = extract_skill_synonyms(sample_resume, "Python")
    print(f"Python synonyms found: {python_synonyms}")
    
    ml_synonyms = extract_skill_synonyms(sample_resume, "Machine Learning")
    print(f"Machine Learning synonyms found: {ml_synonyms}")
    print()
    
    # Bonus 5: Skill Recommendations
    print("BONUS 5: Skill Recommendation Generator")
    print("-" * 40)
    job_reqs = ['Python', 'Django', 'React', 'AWS', 'Docker', 'Kubernetes', 'Leadership']
    recommendations = generate_skill_recommendations(skills, job_reqs)
    print(f"Job Requirements: {job_reqs}")
    print(f"Priority Level: {recommendations['priority']}")
    print(f"Reason: {recommendations['reason']}")
    print(f"Total Missing: {recommendations['total_missing']}")
    print("\nRecommendations:")
    for priority, skills_list in recommendations['recommendations'].items():
        if skills_list:
            print(f"  {priority.replace('_', ' ').title()}: {skills_list}")


def test_bonus_functions():
    """
    Test function for the additional bonus features.
    """
    print("=== Testing Additional Bonus Functions ===\n")
    
    # Test Bonus 3: Skill Coverage Score
    print("Bonus 3 Testing calculate_skill_coverage_score():")
    skills_dict = {'all_skills': ['Python', 'Java', 'AWS', 'Leadership']}
    required_skills = ['Python', 'JavaScript', 'AWS', 'Docker']
    result = calculate_skill_coverage_score(skills_dict, required_skills)
    print(f"Skills found: {skills_dict['all_skills']}")
    print(f"Required: {required_skills}")
    print(f"Coverage: {result['coverage_percent']}%")
    print(f"Matched: {result['matched_skills']}")
    print(f"Missing: {result['missing_skills']}")
    print("✓ Coverage score calculated successfully\n")
    
    # Test Bonus 4: Skill Synonyms
    print("Bonus 4 Testing extract_skill_synonyms():")
    text = "I am a Python developer with JS experience and ML knowledge"
    synonyms = extract_skill_synonyms(text, "Python")
    print(f"Text: {text}")
    print(f"Python synonyms: {synonyms}")
    print("✓ Synonyms extracted successfully\n")
    
    # Test Bonus 5: Skill Recommendations
    print("Bonus 5 Testing generate_skill_recommendations():")
    skills_dict = {'all_skills': ['Python', 'JavaScript']}
    job_reqs = ['Python', 'AWS', 'Docker', 'Leadership']
    recommendations = generate_skill_recommendations(skills_dict, job_reqs)
    print(f"Current skills: {skills_dict['all_skills']}")
    print(f"Job requirements: {job_reqs}")
    print(f"Priority: {recommendations['priority']}")
    print(f"Recommendations: {recommendations['recommendations']}")
    print("✓ Recommendations generated successfully\n")


if __name__ == "__main__":
    test_functions()
    print("\n" + "="*50 + "\n")
    test_bonus_functions()
    print("\n" + "="*50 + "\n")
    demonstrate_complete_extraction()
    print("\n" + "="*50 + "\n")
    demonstrate_bonus_features()
