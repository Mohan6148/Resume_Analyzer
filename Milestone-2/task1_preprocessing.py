"""
Task 1: Text Preprocessing
Milestone 2: NLP & Skill Extraction Tasks

This module contains functions for basic text preprocessing operations:
1. Basic Text Cleaning
2. Tokenization 
3. Stop Words Removal
4. Lemmatization

Author: Mohan Sri Sai Panduri
ID: 22A81A6148@sves.org.in
"""

import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model 'en_core_web_sm' not found. Please install it using:")
    print("python -m spacy download en_core_web_sm")
    nlp = None


def clean_resume_text(text):
    """
    Question 1.1: Basic Text Cleaning (5 points)
    
    Cleans resume text by:
    - Removing email addresses
    - Removing phone numbers  
    - Removing URLs
    - Removing special characters except (+ # - .)
    - Converting to lowercase
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Remove phone numbers (various formats)
    text = re.sub(r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', '', text)
    text = re.sub(r'\+?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove special characters except (+ # - .) and spaces
    text = re.sub(r'[^\w\s+#.-]', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_text(text):
    """
    Question 1.2: Tokenization (5 points)
    
    Tokenizes text using spaCy, handling contractions properly.
    
    Args:
        text (str): Input text to tokenize
        
    Returns:
        list: List of tokens
    """
    if not text or not nlp:
        return []
    
    doc = nlp(text)
    tokens = []
    
    for token in doc:
        # Handle contractions by expanding them
        if token.text in ["I'm", "i'm"]:
            tokens.extend(["I", "am"])
        elif token.text in ["I've", "i've"]:
            tokens.extend(["I", "have"])
        elif token.text in ["don't", "Don't"]:
            tokens.extend(["do", "not"])
        elif token.text in ["can't", "Can't"]:
            tokens.extend(["can", "not"])
        elif token.text in ["won't", "Won't"]:
            tokens.extend(["will", "not"])
        elif token.text in ["n't"]:
            tokens.append("not")
        else:
            tokens.append(token.text)
    
    return tokens


def remove_stop_words(text):
    """
    Question 1.3: Stop Words Removal (5 points)
    
    Removes common stop words while preserving programming language names.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with stop words removed
    """
    if not text or not nlp:
        return text
    
    doc = nlp(text)
    preserved_words = ['c', 'r', 'go', 'd']  # Programming languages to preserve
    
    filtered_tokens = []
    for token in doc:
        # Keep if not a stop word OR if it's a preserved programming language
        if not token.is_stop or token.text.lower() in preserved_words:
            filtered_tokens.append(token.text)
    
    return ' '.join(filtered_tokens)


def lemmatize_text(text):
    """
    Question 1.4: Lemmatization (5 points)
    
    Converts words to their base form using spaCy's lemmatizer.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Lemmatized text
    """
    if not text or not nlp:
        return text
    
    doc = nlp(text)
    lemmatized_tokens = []
    
    for token in doc:
        lemmatized_tokens.append(token.lemma_)
    
    return ' '.join(lemmatized_tokens)


def test_functions():
    """
    Test function to verify all preprocessing functions work correctly.
    """
    print("=== Testing Task 1: Text Preprocessing Functions ===\n")
    
    # Test 1.1: Basic Text Cleaning
    print("1.1 Testing clean_resume_text():")
    test_text1 = """
Contact: john@email.com | Phone: +1-555-0123
Visit: www.johndoe.com
Skills: Python, C++, C#, .NET
"""
    result1 = clean_resume_text(test_text1)
    print(f"Input: {repr(test_text1)}")
    print(f"Output: {result1}")
    print(f"Expected: contact phone visit skills python c++ c .net")
    print(f"Match: {'✓' if result1 == 'contact phone visit skills python c++ c .net' else '✗'}\n")
    
    # Test 1.2: Tokenization
    print("1.2 Testing tokenize_text():")
    test_text2 = "I'm a Python developer. I've worked on ML projects."
    result2 = tokenize_text(test_text2)
    print(f"Input: {test_text2}")
    print(f"Output: {result2}")
    expected2 = ['I', 'am', 'a', 'Python', 'developer', '.', 'I', 'have', 'worked', 'on', 'ML', 'projects', '.']
    print(f"Expected: {expected2}")
    print(f"Match: {'✓' if result2 == expected2 else '✗'}\n")
    
    # Test 1.3: Stop Words Removal
    print("1.3 Testing remove_stop_words():")
    test_text3 = "I have experience in Python and R programming with excellent skills in C and Go"
    result3 = remove_stop_words(test_text3)
    print(f"Input: {test_text3}")
    print(f"Output: {result3}")
    expected3 = "experience Python R programming excellent skills C Go"
    print(f"Expected: {expected3}")
    print(f"Match: {'✓' if result3 == expected3 else '✗'}\n")
    
    # Test 1.4: Lemmatization
    print("1.4 Testing lemmatize_text():")
    test_text4 = "I am working on developing multiple applications using programming languages"
    result4 = lemmatize_text(test_text4)
    print(f"Input: {test_text4}")
    print(f"Output: {result4}")
    expected4 = "I be work on develop multiple application use programming language"
    print(f"Expected: {expected4}")
    print(f"Match: {'✓' if result4 == expected4 else '✗'}\n")


if __name__ == "__main__":
    test_functions()
