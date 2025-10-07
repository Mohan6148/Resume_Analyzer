"""
Task 2: POS Tagging
Milestone 2: NLP & Skill Extraction Tasks

This module contains functions for Part-of-Speech tagging operations:
1. Basic POS Tagging
2. Extract Nouns Only
3. Identify Skill Patterns (Adjective + Noun combinations)

Author: Mohan Sri Sai Panduri
ID: 22A81A6148@sves.org.in
"""

import spacy
import re

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model 'en_core_web_sm' not found. Please install it using:")
    print("python -m spacy download en_core_web_sm")
    nlp = None


def pos_tag_resume(text):
    """
    Question 2.1: Basic POS Tagging (7 points)
    
    Tags each word with its Part of Speech using spaCy.
    
    Args:
        text (str): Input text to tag
        
    Returns:
        list: List of tuples (word, POS_tag)
    """
    if not text or not nlp:
        return []
    
    doc = nlp(text)
    pos_tags = []
    
    for token in doc:
        # Skip whitespace tokens
        if not token.is_space:
            pos_tags.append((token.text, token.pos_))
    
    return pos_tags


def extract_nouns(text):
    """
    Question 2.2: Extract Nouns Only (7 points)
    
    Extracts all NOUN and PROPN (proper nouns) which are potential skills.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of nouns and proper nouns
    """
    if not text or not nlp:
        return []
    
    doc = nlp(text)
    nouns = []
    
    for token in doc:
        # Extract NOUN and PROPN (proper nouns)
        if token.pos_ in ['NOUN', 'PROPN'] and not token.is_space:
            nouns.append(token.text)
    
    return nouns


def find_adj_noun_patterns(text):
    """
    Question 2.3: Identify Skill Patterns (6 points)
    
    Finds all "Adjective + Noun" combinations which often represent skills.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of adjective-noun patterns
    """
    if not text or not nlp:
        return []
    
    doc = nlp(text)
    patterns = []
    
    # Look for ADJ + NOUN patterns
    for i, token in enumerate(doc):
        if token.pos_ == 'ADJ' and not token.is_space:
            # Check if next token is a noun
            next_token = None
            if i + 1 < len(doc):
                next_token = doc[i + 1]
            
            if next_token and next_token.pos_ in ['NOUN', 'PROPN'] and not next_token.is_space:
                pattern = f"{token.text} {next_token.text}"
                patterns.append(pattern)
    
    # Also look for multi-word patterns like "Machine Learning", "Deep Learning"
    # Using regex to find common skill patterns
    skill_patterns = [
        r'\b(?:Machine|Deep|Natural|Artificial|Data|Big|Cloud|Web|Mobile|Front|Back)\s+(?:Learning|Language|Processing|Science|Data|Computing|Development|End|End)\b',
        r'\b(?:Object|Functional|Procedural|Aspect)\s+(?:Oriented|Programming)\b',
        r'\b(?:Full|Back|Front|Mobile|Web|Software|Game|Application)\s+(?:Stack|End|Development)\b',
        r'\b(?:User|Application|Application)\s+(?:Interface|Experience|Programming)\b',
        r'\b(?:Version|Source|Quality)\s+(?:Control|Code|Assurance)\b'
    ]
    
    for pattern in skill_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        patterns.extend(matches)
    
    # Remove duplicates and return
    return list(set(patterns))


def test_functions():
    """
    Test function to verify all POS tagging functions work correctly.
    """
    print("=== Testing Task 2: POS Tagging Functions ===\n")
    
    # Test 2.1: Basic POS Tagging
    print("2.1 Testing pos_tag_resume():")
    test_text1 = "John is an experienced Python developer"
    result1 = pos_tag_resume(test_text1)
    print(f"Input: {test_text1}")
    print(f"Output: {result1}")
    expected1 = [('John', 'PROPN'), ('is', 'AUX'), ('an', 'DET'), 
                ('experienced', 'ADJ'), ('Python', 'PROPN'), ('developer', 'NOUN')]
    print(f"Expected: {expected1}")
    print(f"Match: {'✓' if result1 == expected1 else '✗'}\n")
    
    # Test 2.2: Extract Nouns
    print("2.2 Testing extract_nouns():")
    test_text2 = "Experienced Data Scientist proficient in Machine Learning and Python programming"
    result2 = extract_nouns(test_text2)
    print(f"Input: {test_text2}")
    print(f"Output: {result2}")
    expected2 = ['Data', 'Scientist', 'Machine', 'Learning', 'Python', 'programming']
    print(f"Expected: {expected2}")
    print(f"Match: {'✓' if result2 == expected2 else '✗'}\n")
    
    # Test 2.3: Find Adjective-Noun Patterns
    print("2.3 Testing find_adj_noun_patterns():")
    test_text3 = "Expert in Machine Learning, Deep Learning, and Natural Language Processing"
    result3 = find_adj_noun_patterns(test_text3)
    print(f"Input: {test_text3}")
    print(f"Output: {result3}")
    expected3 = ['Machine Learning', 'Deep Learning', 'Natural Language']
    print(f"Expected: {expected3}")
    # Check if at least the expected patterns are found
    found_patterns = [pattern for pattern in expected3 if any(pattern in r for r in result3)]
    print(f"Expected patterns found: {found_patterns}")
    print(f"Match: {'✓' if len(found_patterns) >= 2 else '✗'}\n")


def demonstrate_pos_analysis():
    """
    Additional demonstration of POS tagging capabilities.
    """
    print("=== POS Tagging Demonstration ===\n")
    
    sample_resume_text = """
    I am a Senior Python Developer with 5 years of experience.
    I have worked on Machine Learning projects using TensorFlow and PyTorch.
    My skills include Data Science, Web Development, and Cloud Computing.
    I am proficient in JavaScript, React, and Node.js frameworks.
    """
    
    print("Sample Resume Text:")
    print(sample_resume_text)
    
    # POS Tagging
    print("POS Tags:")
    pos_tags = pos_tag_resume(sample_resume_text)
    for word, tag in pos_tags:
        print(f"  {word}: {tag}")
    
    print("\nExtracted Nouns (Potential Skills):")
    nouns = extract_nouns(sample_resume_text)
    for noun in nouns:
        print(f"  • {noun}")
    
    print("\nAdjective-Noun Patterns (Skill Combinations):")
    patterns = find_adj_noun_patterns(sample_resume_text)
    for pattern in patterns:
        print(f"  • {pattern}")


if __name__ == "__main__":
    test_functions()
    print("\n" + "="*50 + "\n")
    demonstrate_pos_analysis()
