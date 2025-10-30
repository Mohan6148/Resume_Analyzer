"""
Skill Gap Analysis & Similarity Matching
Uses S-BERT embeddings and cosine similarity to identify skill gaps
"""

try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None

from typing import Dict, List, Tuple

try:
    import streamlit as st
except Exception:
    class _StFallback:
        def error(self, msg):
            print("[STREAMLIT ERROR]", msg)
        def warning(self, msg):
            print("[STREAMLIT WARNING]", msg)
        def info(self, msg):
            print("[STREAMLIT INFO]", msg)
        def success(self, msg):
            print("[STREAMLIT SUCCESS]", msg)

    st = _StFallback()

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    cosine_similarity = None
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SIMILARITY_THRESHOLD, SBERT_MODEL


class SkillGapAnalyzer:
    """Analyzes skill gaps between resume and job description"""
    
    def __init__(self, model_name: str = None):
        """Initialize BERT model for embeddings"""
        self.model_name = model_name or SBERT_MODEL
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Sentence-BERT model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            st.success(f"✅ Loaded {self.model_name}")
        except Exception as e:
            st.error(f"Failed to load model {self.model_name}: {str(e)}")
            st.info(f"Install with: pip install sentence-transformers")
    
    def analyze_gap(self, resume_skills: List[str], jd_skills: List[str], 
                   threshold: float = SIMILARITY_THRESHOLD) -> Dict:
        """
        Analyze skill gap between resume and job description
        
        Args:
            resume_skills: List of skills found in resume
            jd_skills: List of skills from job description
            threshold: Similarity threshold for matching
            
        Returns:
            Dictionary with gap analysis results
        """
        # Deduplicate and normalize input skills first
        resume_skills_clean = list(set([s.lower().strip() for s in resume_skills if s]))
        jd_skills_clean = list(set([s.lower().strip() for s in jd_skills if s]))

        # If Sentence-BERT model not available, fall back to TF-IDF cosine similarity
        similarity_matrix = None
        if not self.model:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

                if not resume_skills_clean or not jd_skills_clean:
                    return {'error': 'No skills found'}

                # Vectorize the skill strings (each skill is a document)
                vectorizer = TfidfVectorizer().fit_transform(resume_skills_clean + jd_skills_clean)
                resume_vecs = vectorizer[: len(resume_skills_clean) ]
                jd_vecs = vectorizer[ len(resume_skills_clean): ]

                similarity_matrix = _cos_sim(jd_vecs, resume_vecs)
            except Exception as e:
                st.error(f"Fallback TF-IDF similarity failed: {str(e)}")
                return {}
        
        if not resume_skills_clean or not jd_skills_clean:
            return {'error': 'No skills found'}
        
        # If we have a model, get embeddings and compute similarity matrix
        if self.model and similarity_matrix is None:
            try:
                resume_embeddings = self.model.encode(resume_skills_clean, show_progress_bar=False)
                jd_embeddings = self.model.encode(jd_skills_clean, show_progress_bar=False)
                similarity_matrix = cosine_similarity(jd_embeddings, resume_embeddings)
            except Exception as e:
                st.error(f"Encoding error: {str(e)}")
                return {}
        
        # Find best matches for each JD skill
        matches = []
        for i, jd_skill in enumerate(jd_skills_clean):
            best_match_idx = np.argmax(similarity_matrix[i])
            best_match_score = float(similarity_matrix[i][best_match_idx])
            best_match_skill = resume_skills_clean[best_match_idx]
            
            matches.append({
                'jd_skill': jd_skill,
                'resume_match': best_match_skill,
                'similarity': best_match_score,
                'match_status': 'match' if best_match_score >= threshold else 'gap'
            })
        
        # Calculate overall match percentage
        match_count = sum(1 for m in matches if m['match_status'] == 'match')
        overall_match = (match_count / len(matches) * 100) if matches else 0
        
        # Identify gaps (missing skills)
        gaps = [m for m in matches if m['match_status'] == 'gap']
        gaps_sorted = sorted(gaps, key=lambda x: x['similarity'])
        
        # Identify high-priority gaps (very low similarity)
        high_priority_gaps = [g for g in gaps if g['similarity'] < 0.3]
        
        return {
            'overall_match': overall_match,
            'total_jd_skills': len(jd_skills_clean),
            'matched_skills': match_count,
            'gaps': gaps_sorted,
            'high_priority_gaps': high_priority_gaps,
            'similarity_matrix': similarity_matrix.tolist(),
            'jd_skills': jd_skills_clean,
            'resume_skills': resume_skills_clean,
            'matches': matches
        }
    
    def rank_gaps_by_importance(self, gap_analysis: Dict) -> List[Dict]:
        """Rank skill gaps by importance and missing priority"""
        gaps = gap_analysis.get('gaps', [])
        
        # Rank by similarity (lower = more critical gap)
        ranked_gaps = sorted(gaps, key=lambda x: x['similarity'])
        
        # Add priority scoring
        for i, gap in enumerate(ranked_gaps):
            gap['priority_score'] = (1 - gap['similarity']) * 100
            gap['rank'] = i + 1
        
        return ranked_gaps
    
    def get_category_analysis(self, gap_analysis: Dict, 
                             skill_categories: Dict) -> Dict:
        """Analyze gaps by skill category"""
        if 'matches' not in gap_analysis:
            return {}
        
        category_stats = {}
        
        for match in gap_analysis['matches']:
            jd_skill = match['jd_skill']
            
            # Determine category for JD skill
            category = self._get_category_for_skill(jd_skill, skill_categories)
            
            if category:
                if category not in category_stats:
                    category_stats[category] = {
                        'total': 0,
                        'matched': 0,
                        'gaps': []
                    }
                
                category_stats[category]['total'] += 1
                
                if match['match_status'] == 'match':
                    category_stats[category]['matched'] += 1
                else:
                    category_stats[category]['gaps'].append(jd_skill)
        
        # Calculate match percentages per category
        for category in category_stats:
            stats = category_stats[category]
            stats['match_percentage'] = (stats['matched'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        return category_stats
    
    def _get_category_for_skill(self, skill: str, skill_categories: Dict) -> str:
        """Determine category for a skill"""
        skill_lower = skill.lower()
        
        for category, skills in skill_categories.items():
            if skill_lower in [s.lower() for s in skills]:
                return category
        
        return 'Other'



