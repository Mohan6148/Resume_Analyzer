"""
Run a minimal end-to-end pipeline (non-Streamlit) for the SkillGapAI project.

This script reads the sample files in `sample_data/`, runs ingestion, skill extraction,
similarity analysis (SBERT if available, TF-IDF fallback otherwise), and writes a CSV
report to `outputs/skill_gap_analysis.csv`.

Usage:
    python run_pipeline.py

The script is defensive: if heavy models are not installed it will fall back to
lightweight heuristics so you can test the flow without installing everything.
"""
import io
import os
import sys
from pathlib import Path
import json

# Make sure backend package is importable
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from backend.data_ingestion import DataIngestion
from backend.skill_extraction import SkillExtractor
from backend.gap_analysis import SkillGapAnalyzer
from backend.export_manager import ExportManager
from config import SKILL_CATEGORIES, SBERT_MODEL

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def make_file_like(path: Path):
    """Return a simple file-like object expected by DataIngestion for txt files."""
    b = path.read_bytes()
    bio = io.BytesIO(b)
    bio.name = path.name
    # Provide read() for parse_txt
    bio.read = bio.getvalue
    return bio


def fallback_extract_skills(text: str):
    """Simple fallback extractor using SKILL_CATEGORIES keyword matching."""
    import re
    counts = {}
    contexts = {}
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    for category, skills in SKILL_CATEGORIES.items():
        for skill in skills:
            pattern = re.compile(rf"\b{re.escape(skill)}\b", re.IGNORECASE)
            matches = pattern.findall(text)
            if matches:
                counts[skill.lower()] = len(matches)
                # find a sentence for context
                for sent in sentences:
                    if pattern.search(sent):
                        contexts.setdefault(skill.lower(), []).append(sent)
                        break

    result = {
        'skill_counts': counts,
        'skill_contexts': contexts,
        'all_skills': counts
    }
    return result


def fallback_similarity(resume_skills, jd_skills):
    """Compute similarity matrix using TF-IDF on skill phrases as fallback."""
    if not resume_skills or not jd_skills:
        return {}

    corpus = resume_skills + jd_skills
    vectorizer = TfidfVectorizer().fit_transform(corpus)
    resume_vecs = vectorizer[: len(resume_skills) ]
    jd_vecs = vectorizer[ len(resume_skills): ]

    # We want jd x resume similarity
    sim = cosine_similarity(jd_vecs, resume_vecs)

    matches = []
    for i, jd in enumerate(jd_skills):
        best_idx = sim[i].argmax()
        best_score = float(sim[i][best_idx])
        matches.append({
            'jd_skill': jd,
            'resume_match': resume_skills[best_idx] if resume_skills else '',
            'similarity': best_score,
            'match_status': 'match' if best_score >= 0.6 else 'gap'
        })

    overall_match = sum(1 for m in matches if m['match_status'] == 'match') / len(matches) * 100 if matches else 0

    return {
        'overall_match': overall_match,
        'total_jd_skills': len(jd_skills),
        'matched_skills': sum(1 for m in matches if m['match_status'] == 'match'),
        'gaps': [m for m in matches if m['match_status'] == 'gap'],
        'high_priority_gaps': [m for m in matches if m['similarity'] < 0.3],
        'similarity_matrix': sim.tolist(),
        'jd_skills': jd_skills,
        'resume_skills': resume_skills,
        'matches': matches
    }


def ensure_outputs_dir():
    out = ROOT / 'outputs'
    out.mkdir(exist_ok=True)
    return out


def run_on_samples():
    sample_resume = ROOT / 'sample_data' / 'sample_resume.txt'
    sample_jd = ROOT / 'sample_data' / 'sample_job_description.txt'

    if not sample_resume.exists() or not sample_jd.exists():
        print("Sample files not found in sample_data/. Place sample_resume.txt and sample_job_description.txt there.")
        return

    ingestion = DataIngestion()

    resume_file = make_file_like(sample_resume)
    jd_file = make_file_like(sample_jd)

    raw_r, cleaned_r, err_r = ingestion.parse_document(resume_file, {'name': resume_file.name})
    raw_j, cleaned_j, err_j = ingestion.parse_document(jd_file, {'name': jd_file.name})

    if err_r or err_j:
        print('Error parsing files:', err_r, err_j)
        return

    # Skill extraction
    extractor = SkillExtractor()

    if extractor.model_loaded:
        resume_skills_res = extractor.extract_skills(cleaned_r)
        jd_skills_res = extractor.extract_skills(cleaned_j)
    else:
        print('spaCy model not available; using fallback keyword extractor')
        resume_skills_res = fallback_extract_skills(cleaned_r)
        jd_skills_res = fallback_extract_skills(cleaned_j)

    resume_skills = list(resume_skills_res.get('skill_counts', {}).keys())
    jd_skills = list(jd_skills_res.get('skill_counts', {}).keys())

    if not resume_skills:
        print('No resume skills found with configured extractor; attempting simple tokenization.')
        resume_skills = [w.strip().lower() for w in cleaned_r.split()[:100] if len(w) > 2]

    if not jd_skills:
        print('No JD skills found with configured extractor; attempting simple tokenization.')
        jd_skills = [w.strip().lower() for w in cleaned_j.split()[:100] if len(w) > 2]

    # Similarity / gap analysis
    analyzer = None
    try:
        analyzer = SkillGapAnalyzer()
    except Exception:
        analyzer = None

    if analyzer and analyzer.model:
        gap_analysis = analyzer.analyze_gap(resume_skills, jd_skills)
    else:
        print('SBERT model not available; using TF-IDF fallback for similarity')
        gap_analysis = fallback_similarity(resume_skills, jd_skills)

    # Export
    out_dir = ensure_outputs_dir()
    csv_path = out_dir / 'skill_gap_analysis.csv'
    exporter = ExportManager()
    exporter.export_to_csv(gap_analysis, str(csv_path))

    print('\n=== Analysis Summary ===')
    print(f"Overall Match: {gap_analysis.get('overall_match', 0):.1f}%")
    print(f"Total JD Skills: {gap_analysis.get('total_jd_skills', 0)}")
    print(f"Matched Skills: {gap_analysis.get('matched_skills', 0)}")
    print(f"Gaps: {len(gap_analysis.get('gaps', []))}")
    print(f"CSV report written to: {csv_path}\n")


if __name__ == '__main__':
    run_on_samples()
