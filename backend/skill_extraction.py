"""
Skill Extraction using NLP
Uses spaCy + custom NER to extract technical and soft skills
"""

try:
    import spacy
except Exception:
    spacy = None

import re
from typing import Dict, List, Tuple

try:
    import streamlit as st
except Exception:
    class _StFallback:
        def warning(self, msg):
            print("[STREAMLIT WARNING]", msg)
    st = _StFallback()

from collections import Counter

try:
    import pandas as pd
except Exception:
    pd = None

# Import configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SKILL_CATEGORIES


class SkillExtractor:
    """Extracts skills from text using NLP and custom NER"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize spaCy model"""
        try:
            self.nlp = spacy.load(model_name)
            self.model_loaded = True
        except Exception as e:
            st.warning(f"spaCy model not loaded: {str(e)}. Install with: python -m spacy download en_core_web_sm")
            self.model_loaded = False
        
        # Compile skill patterns
        self._build_skill_patterns()
    
    def _build_skill_patterns(self):
        """Build regex patterns for skill matching"""
        # Flatten all skills into a single list for matching
        # store canonical lowercase skills for consistent matching
        self.all_skills = []
        for category, skills in SKILL_CATEGORIES.items():
            self.all_skills.extend([s.lower() for s in skills])
        
        # Create regex patterns
        self.skill_patterns = {}
        for skill in self.all_skills:
            # Case insensitive pattern; use word boundaries to avoid partial matches
            pattern = rf'\b{re.escape(skill)}\b'
            self.skill_patterns[skill] = re.compile(pattern, re.IGNORECASE)
    
    def extract_skills(self, text: str, confidence_threshold: float = 0.6) -> Dict:
        """
        Extract skills from text using multiple methods
        
        Returns:
            Dictionary with extracted skills grouped by category
        """
        results = {
            'technical': [],
            'soft': [],
            'tools': [],
            'certifications': [],
            'categories': {}
        }

        if not text:
            return results

        # Process text with spaCy if available
        doc = None
        if self.model_loaded and self.nlp is not None:
            try:
                doc = self.nlp(text)
            except Exception:
                doc = None

        # Extract using pattern matching (always run)
        matched_skills = self._extract_by_patterns(text)

        # Extract using POS tagging only if spaCy doc is available
        pos_skills = self._extract_by_pos(doc) if doc is not None else []

        # Extract named entities only if spaCy doc is available
        ner_skills = self._extract_ner(doc) if doc is not None else []
        
        # Combine all methods into a single mapping: skill -> list of occurrences
        all_found_skills = {}

        # Start with pattern matches (these already provide occurrences lists)
        for skill, occurrences in matched_skills.items():
            key = skill.lower()
            all_found_skills.setdefault(key, [])
            if isinstance(occurrences, list):
                all_found_skills[key].extend(occurrences)
            else:
                all_found_skills[key].append(occurrences)

        # Add POS-extracted skills (list of skill tokens)
        for skill in pos_skills:
            key = skill.lower()
            all_found_skills.setdefault(key, [])
            all_found_skills[key].append(skill)

        # Add NER-extracted skills
        for skill in ner_skills:
            key = skill.lower()
            all_found_skills.setdefault(key, [])
            all_found_skills[key].append(skill)

        # Categorize skills by the canonical skill name (the dict keys)
        for skill in list(all_found_skills.keys()):
            category = self._categorize_skill(skill)
            if category:
                results['categories'].setdefault(category, []).append(skill)
        
        # Get skill contexts (sentences where skills appear)
        skill_contexts = self._get_skill_contexts(text, all_found_skills)
        
        results['all_skills'] = all_found_skills
        results['skill_contexts'] = skill_contexts
        results['skill_counts'] = {skill: len(occurrences) for skill, occurrences in all_found_skills.items()}
        
        return results
    
    def _extract_by_patterns(self, text: str) -> Dict:
        """Extract skills using pre-defined patterns"""
        found_skills = {}
        
        for skill, pattern in self.skill_patterns.items():
            matches = pattern.findall(text)
            if matches:
                found_skills[skill] = matches
        
        return found_skills
    
    def _extract_by_pos(self, doc) -> List[str]:
        """Extract skills using POS tagging"""
        skills = []
        
        # Look for noun phrases that might be skills
        if doc is None:
            return skills

        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:
                # Check if token could be a skill
                lemma = token.lemma_.lower()
                if lemma in self.all_skills:
                    skills.append(lemma)
        
        return skills
    
    def _extract_ner(self, doc) -> List[str]:
        """Extract named entities that might be skills"""
        skills = []
        if doc is None:
            return skills

        # Only consider certain entity labels, but prefer matching against known skills.
        for ent in doc.ents:
            if ent.label_ in ['PRODUCT', 'TECH', 'SKILL']:
                entity_text = ent.text.lower()
                # Check against skill list and append the canonical skill (not the entity text)
                for skill in self.all_skills:
                    # match when skill appears inside the entity or vice-versa
                    if skill in entity_text or entity_text in skill:
                        # append canonical skill only
                        skills.append(skill)
                        break

        # As a defensive measure, also check PERSON/ORG/GPE entities for embedded skills
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'GPE', 'PERSON']:
                entity_text = ent.text.lower()
                for skill in self.all_skills:
                    if skill in entity_text:
                        skills.append(skill)
                        break
        
        return skills
    
    def _categorize_skill(self, skill: str) -> str:
        """Categorize a skill into predefined categories"""
        skill_lower = skill.lower()
        
        for category, skills in SKILL_CATEGORIES.items():
            if skill_lower in [s.lower() for s in skills]:
                return category
        
        return None
    
    def _get_skill_contexts(self, text: str, found_skills: Dict) -> Dict:
        """Get context sentences for each found skill"""
        contexts = {}
        
        for skill, occurrences in found_skills.items():
            contexts[skill] = []
            for occurrence in occurrences:
                # Find sentence containing this skill
                sentences = re.split(r'[.!?]+', text)
                for sentence in sentences:
                    if skill.lower() in sentence.lower():
                        contexts[skill].append(sentence.strip())
                        break
        
        return contexts
    
    def get_skill_frequency(self, extracted_skills: Dict) -> pd.DataFrame:
        """Get frequency distribution of skills"""
        skill_counts = extracted_skills.get('skill_counts', {})
        
        if not skill_counts:
            return pd.DataFrame()
        
        df = pd.DataFrame([
            {'skill': skill, 'frequency': count}
            for skill, count in skill_counts.items()
        ])
        
        df = df.sort_values('frequency', ascending=False)
        return df
    
    def filter_by_confidence(self, extracted_skills: Dict, threshold: float) -> Dict:
        """Filter skills by confidence threshold (in this case, frequency)"""
        filtered = {
            'technical': [],
            'soft': [],
            'tools': [],
            'certifications': []
        }
        
        skill_counts = extracted_skills.get('skill_counts', {})
        
        for skill, frequency in skill_counts.items():
            # Use frequency as proxy for confidence
            confidence = min(1.0, frequency / 10.0)  # Normalize to 0-1
            
            if confidence >= threshold:
                category = self._categorize_skill(skill)
                if category:
                    filtered.setdefault(category.lower(), []).append({
                        'skill': skill,
                        'frequency': frequency,
                        'confidence': confidence
                    })
        
        return filtered

