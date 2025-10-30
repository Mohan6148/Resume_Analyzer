# Resume_Analyzer
# 🧠 SkillGapAI – Analyzing Resume and Job Post for Skill Gap

## 📌 Overview
**SkillGapAI** is an AI-powered system designed to identify and analyze skill gaps between a **candidate’s resume** and a **job description**.  
The application extracts, compares, and visualizes technical and soft skills, enabling both recruiters and job seekers to understand where upskilling is needed.

## 🎯 Objectives
- Automate skill extraction from resumes and job descriptions using **Natural Language Processing (NLP)**.  
- Identify **missing, matched, and unique skills** efficiently.  
- Provide **visual insights** into the skill gap through interactive dashboards.  
- Allow easy **export of analytical reports** (CSV / PDF formats).  

## ⚙️ Features
✅ Resume and Job Description text extraction (PDF / DOCX / TXT)  
✅ Skill extraction using **spaCy** and **Sentence Transformers**  
✅ Skill gap analysis using **semantic similarity**  
✅ Visualization with **Plotly charts** and **Streamlit UI**  
✅ Export results to **CSV** and **PDF**  
✅ Clean and modern interface with **custom CSS design**

SkillGapAI/
│
├── backend/
│ ├── data_ingestion.py
│ ├── skill_extraction.py
│ ├── gap_analysis.py
│ ├── visualizations.py
│ ├── export_manager.py
│ └── init.py
│
├── config.py
├── main.py
├── requirements.txt
└── README.md

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mohan6148/Resume_Analyzer
   cd Resume_Analyzer

2. **Install dependencies**
    pip install -r requirements.txt

3. **Run the Streamlit app**
    streamlit run main.py

4.**Upload resume and job description files, then view analysis results on the dashboard**

📄 License
This project is open-source under the MIT License.
