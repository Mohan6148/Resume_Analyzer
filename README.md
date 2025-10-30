## SkillGapAI — Quick Project Outline

SkillGapAI analyzes a resume against a job description and identifies skill matches and gaps.

1. Upload resume + job description (PDF/DOCX/TXT)
2. Extract skills (pattern matching + spaCy NER if available)
3. Encode skills and compute similarity (SBERT or TF-IDF fallback)
4. Identify and rank gaps
5. Visualize results and export CSV/PDF reports

Quick start
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # optional but recommended
streamlit run app.py
```

Sample files are in `sample_data/`.

Built with Python and Streamlit.
