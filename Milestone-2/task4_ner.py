"""
Task 4: Named Entity Recognition
Milestone 2: NLP & Skill Extraction Tasks

This module contains functions for Named Entity Recognition operations:
1. Extract Named Entities
2. Annotate Training Data

Author: Mohan Sri Sai Panduri
ID: 22A81A6148@sves.org.in
"""

import spacy
from typing import List, Tuple

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model 'en_core_web_sm' not found. Please install it using:")
    print("python -m spacy download en_core_web_sm")
    nlp = None


def extract_entities(text):
    """
    Question 4.1: Extract Named Entities (8 points)
    
    Uses spaCy's NER to extract entities with focus on ORG, PERSON, GPE, PRODUCT.
    
    Args:
        text (str): Input text to extract entities from
        
    Returns:
        list: List of tuples (entity_text, entity_label)
    """
    if not text or not nlp:
        return []
    
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        # Focus on specific entity types as requested
        if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT']:
            entities.append((ent.text, ent.label_))
    
    return entities


# Question 4.2: Annotate Training Data (7 points)
# Given sentences for manual annotation
sentences = [
    "Python developer with 5 years of experience",
    "Expert in Machine Learning and Data Science",
    "Proficient in TensorFlow and PyTorch frameworks",
    "Strong SQL and MongoDB database skills",
    "Excellent communication and leadership abilities"
]

def annotate_training_data():
    """
    Manually annotate the given sentences with skill entities in spaCy format.
    Returns the training data with character positions for each skill.
    """
    
    # Manual annotation with character counting
    TRAIN_DATA = [
        # Sentence 1: "Python developer with 5 years of experience"
        # Python: positions 0-6 (0 to 6)
        ("Python developer with 5 years of experience", 
         {"entities": [(0, 6, "SKILL")]}),
        
        # Sentence 2: "Expert in Machine Learning and Data Science"
        # Machine Learning: positions 10-26 (10 to 26)
        # Data Science: positions 31-43 (31 to 43)
        ("Expert in Machine Learning and Data Science",
         {"entities": [(10, 26, "SKILL"), (31, 43, "SKILL")]}),
        
        # Sentence 3: "Proficient in TensorFlow and PyTorch frameworks"
        # TensorFlow: positions 13-23 (13 to 23)
        # PyTorch: positions 28-35 (28 to 35)
        ("Proficient in TensorFlow and PyTorch frameworks",
         {"entities": [(13, 23, "SKILL"), (28, 35, "SKILL")]}),
        
        # Sentence 4: "Strong SQL and MongoDB database skills"
        # SQL: positions 7-10 (7 to 10)
        # MongoDB: positions 15-22 (15 to 22)
        ("Strong SQL and MongoDB database skills",
         {"entities": [(7, 10, "SKILL"), (15, 22, "SKILL")]}),
        
        # Sentence 5: "Excellent communication and leadership abilities"
        # communication: positions 9-22 (9 to 22)
        # leadership: positions 27-37 (27 to 37)
        ("Excellent communication and leadership abilities",
         {"entities": [(9, 22, "SKILL"), (27, 37, "SKILL")]})
    ]
    
    return TRAIN_DATA


def verify_annotations():
    """
    Verify the character positions in the training data annotations.
    """
    train_data = annotate_training_data()
    
    print("=== Training Data Annotation Verification ===\n")
    
    for i, (text, entities_dict) in enumerate(train_data):
        print(f"Sentence {i+1}: {text}")
        print(f"Length: {len(text)} characters")
        
        for start, end, label in entities_dict["entities"]:
            extracted_text = text[start:end]
            print(f"  {label}: '{extracted_text}' (positions {start}-{end})")
        
        print()


def demonstrate_entity_extraction():
    """
    Demonstrate entity extraction on sample text.
    """
    print("=== Named Entity Recognition Demonstration ===\n")
    
    sample_text = "John worked at Google and Microsoft in New York. He used TensorFlow and Python."
    
    print(f"Sample Text: {sample_text}")
    print()
    
    entities = extract_entities(sample_text)
    
    print("Extracted Entities:")
    for entity_text, entity_label in entities:
        print(f"  {entity_text}: {entity_label}")
    
    print()
    print("Expected Output:")
    print("  John: PERSON")
    print("  Google: ORG")
    print("  Microsoft: ORG")
    print("  New York: GPE")
    print("  TensorFlow: PRODUCT")
    print("  Python: PRODUCT")


def test_functions():
    """
    Test function to verify all NER functions work correctly.
    """
    print("=== Testing Task 4: Named Entity Recognition Functions ===\n")
    
    # Test 4.1: Extract Named Entities
    print("4.1 Testing extract_entities():")
    test_text = "John worked at Google and Microsoft in New York. He used TensorFlow and Python."
    result = extract_entities(test_text)
    print(f"Input: {test_text}")
    print(f"Output: {result}")
    expected = [('John', 'PERSON'), ('Google', 'ORG'), ('Microsoft', 'ORG'), 
               ('New York', 'GPE'), ('TensorFlow', 'PRODUCT'), ('Python', 'PRODUCT')]
    print(f"Expected: {expected}")
    print(f"Match: {'✓' if result == expected else '✗'}\n")
    
    # Test 4.2: Training Data Annotation
    print("4.2 Testing annotate_training_data():")
    train_data = annotate_training_data()
    print(f"Number of annotated sentences: {len(train_data)}")
    print("Training data created successfully ✓\n")
    
    # Display training data
    print("Training Data:")
    for i, (text, entities_dict) in enumerate(train_data):
        print(f"  {i+1}. {text}")
        for start, end, label in entities_dict["entities"]:
            print(f"     {label}: '{text[start:end]}' ({start}-{end})")


def show_character_counting_examples():
    """
    Show detailed character counting examples for annotation.
    """
    print("=== Character Counting Examples ===\n")
    
    examples = [
        ("Python developer with 5 years of experience", "Python"),
        ("Expert in Machine Learning and Data Science", "Machine Learning"),
        ("Proficient in TensorFlow and PyTorch frameworks", "TensorFlow"),
        ("Strong SQL and MongoDB database skills", "MongoDB"),
        ("Excellent communication and leadership abilities", "communication")
    ]
    
    for text, target_skill in examples:
        print(f"Text: '{text}'")
        print(f"Target: '{target_skill}'")
        
        # Find the skill in the text
        start = text.find(target_skill)
        end = start + len(target_skill)
        
        print(f"Position: {start}-{end}")
        print(f"Verification: '{text[start:end]}'")
        print()


if __name__ == "__main__":
    test_functions()
    print("\n" + "="*50 + "\n")
    demonstrate_entity_extraction()
    print("\n" + "="*50 + "\n")
    verify_annotations()
    print("\n" + "="*50 + "\n")
    show_character_counting_examples()
