import streamlit as st
import pandas as pd
from src.pipeline import run_pipeline
from datetime import datetime

st.set_page_config(page_title="AI Skill Gap Analyzer - Document Processing", layout="wide")

# Sidebar: Information and Status
with st.sidebar:
    st.header("Information")
    st.markdown("""
    **Supported Formats:**
    - PDF files
    - DOCX files
    - TXT files
    
    **File Size Limit:** 10MB per file
    
    **Processing Steps:**
    1. File upload & validation
    2. Text extraction
    3. Text cleaning & preprocessing
    4. Results display & download
    """)
    st.header("Status")
    if st.session_state.get('results'):
        total = len(st.session_state['results'])
        success = sum(1 for r in st.session_state['results'] if r['success'])
        st.success(f"Processed: {success}/{total} documents")
    else:
        st.info("No documents processed yet")





# Premium Dashboard UI: Sidebar, Header, Stepper, Cards
st.markdown("""
<style>
body, .stApp { background: #181c24 !important; }
.sidebar-content { background: #1a2233; border-radius: 0 20px 20px 0; padding: 2em 1.5em 2em 1.5em; min-height: 100vh; }
.sidebar-logo { text-align: center; margin-bottom: 2em; }
.sidebar-logo img { width: 60px; border-radius: 12px; box-shadow: 0 2px 8px #00f2fe55; }
.main-title {
    font-size: 2.8rem;
    font-weight: bold;
    background: linear-gradient(90deg, #00f2fe 0%, #4facfe 50%, #43e97b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2em;
    letter-spacing: 1.5px;
    text-align: center;
}
.subtitle {
    font-size: 1.2rem;
    color: #b2becd;
    margin-bottom: 2.2em;
    text-align: center;
}
.stepper {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 2em;
    margin-top: 1em;
}
.step {
    flex: 1;
    text-align: center;
    color: #bbb;
    font-weight: 500;
    position: relative;
    font-size: 1.1rem;
    z-index: 1;
}
.step-circle {
    display: inline-block;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #00f2fe 0%, #43e97b 100%);
    color: #181c24;
    font-weight: bold;
    font-size: 1.1rem;
    line-height: 32px;
    box-shadow: 0 0 12px #00f2fe99;
    margin-bottom: 0.3em;
}
.step.active .step-circle {
    background: linear-gradient(135deg, #43e97b 0%, #00f2fe 100%);
    color: #fff;
    box-shadow: 0 0 18px #43e97b99;
}
.step:not(:last-child)::after {
    content: '';
    position: absolute;
    top: 16px;
    right: -50%;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, #00f2fe 10%, #43e97b 90%);
    z-index: 0;
    border-radius: 2px;
}
.glow-card {
    background: rgba(24, 28, 36, 0.98);
    border-radius: 18px;
    box-shadow: 0 8px 32px 0 #00f2fe33, 0 1.5px 8px 0 #43e97b22;
    border: 1.5px solid #232b3b;
    padding: 2em 1.5em 1.5em 1.5em;
    margin-bottom: 2em;
    color: #eaf6fb;
}
.upload-label {
    font-size: 1.1rem;
    font-weight: bold;
    color: #43e97b;
    margin-bottom: 0.5em;
}
.stat-box {
    background: linear-gradient(90deg, #23272f 60%, #2e2e38 100%);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    color: #fff;
    font-size: 1.1rem;
    box-shadow: 0 2px 8px #00f2fe33;
}
.process-btn {
    background: linear-gradient(90deg, #00f2fe 10%, #43e97b 90%);
    color: #181c24 !important;
    font-weight: bold;
    border: none;
    border-radius: 10px;
    padding: 0.9em 2.7em;
    font-size: 1.18rem;
    margin-top: 1.7em;
    box-shadow: 0 2px 12px #00f2fe55;
    transition: 0.2s;
}
.process-btn:hover {
    filter: brightness(1.1);
    box-shadow: 0 4px 16px #43e97b99;
}
.expander-header {
    font-weight: bold;
    color: #43e97b;
    font-size: 1.1rem;
}
</style>
<div class="main-title"><img src="https://img.icons8.com/fluency/48/ai.png" style="vertical-align:middle;margin-right:12px;">AI Skill Gap Analyzer</div>
<div class="subtitle">Empowering you to bridge the gap between your resume and your dream job!</div>
<div class="stepper">
    <div class="step active"><span class="step-circle">1</span><br>Upload</div>
    <div class="step"><span class="step-circle">2</span><br>Analyze</div>
    <div class="step"><span class="step-circle">3</span><br>Results</div>
    <div class="step"><span class="step-circle">4</span><br>Download</div>
</div>
""", unsafe_allow_html=True)




# Upload Section with Glowing Cards and Icons
st.markdown('<div class="section-header" style="color:#43e97b;">1. Upload Documents</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.markdown('<span class="upload-label">📄 Resume Upload</span>', unsafe_allow_html=True)
    resume_files = st.file_uploader(
        "Choose Resume files (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"],
        accept_multiple_files=True, key="resume_uploader")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.markdown('<span class="upload-label">💼 Job Description Upload</span>', unsafe_allow_html=True)
    job_files = st.file_uploader(
        "Choose Job Description files (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"],
        accept_multiple_files=True, key="job_uploader")
    st.markdown('</div>', unsafe_allow_html=True)

all_files = []
if resume_files:
    for file in resume_files:
        all_files.append({
            'bytes': file.getvalue(),
            'type': file.name.split('.')[-1].lower(),
            'name': file.name,
            'doc_type': 'resume',
            'upload_time': datetime.now()
        })
if job_files:
    for file in job_files:
        all_files.append({
            'bytes': file.getvalue(),
            'type': file.name.split('.')[-1].lower(),
            'name': file.name,
            'doc_type': 'job_description',
            'upload_time': datetime.now()
        })




# Process Button (centered, glowing)
if all_files:
    st.markdown('<div class="section-header" style="color:#43e97b;">2. Process Documents</div>', unsafe_allow_html=True)
    col_btn = st.columns([1,2,1])
    with col_btn[1]:
        st.markdown('<div class="glow-card" style="text-align:center;">', unsafe_allow_html=True)
        process = st.button("🚀 Analyze Documents", key="process_btn", help="Click to process all uploaded documents.", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    if process:
        with st.spinner("Processing documents..."):
            results = run_pipeline(all_files)
            st.session_state['results'] = results

# Results Section
if 'results' in st.session_state:
    results = st.session_state['results']
    st.markdown('<div class="section-header" style="color:#43e97b;">3. Processing Results</div>', unsafe_allow_html=True)
    total = len(results)
    success = sum(1 for r in results if r['success'])
    failed = total - success
    st.markdown(f"<div class='stat-box'><b>Total:</b> {total} &nbsp;&nbsp; <b>Success:</b> {success} &nbsp;&nbsp; <b>Failed:</b> {failed} &nbsp;&nbsp; <b>Success Rate:</b> {success/total*100 if total else 0:.1f}%</div>", unsafe_allow_html=True)

    for r in results:
        with st.expander(f"<span class='expander-header'>{r['file']['name']}</span> ({r['file']['doc_type']}) - {'✅' if r['success'] else '❌'}", expanded=False):
            if r['success']:
                st.markdown("<span style='color:#43e97b;font-weight:bold;'>Cleaned Text Preview:</span>", unsafe_allow_html=True)
                preview = r['cleaned'][:1000] + ("..." if len(r['cleaned']) > 1000 else "")
                st.text_area("Cleaned Text", preview, height=200, disabled=True)
            else:
                st.error(f"Error: {r['error']}")

    # Download options
    st.markdown('<div class="section-header" style="color:#43e97b;">4. Download Processed Data</div>', unsafe_allow_html=True)
    df = pd.DataFrame([
        {
            'filename': r['file']['name'],
            'doc_type': r['file']['doc_type'],
            'cleaned_text': r['cleaned'] if r['success'] else '',
            'error': r['error'] if not r['success'] else ''
        } for r in results
    ])
    col_dl = st.columns(2)
    with col_dl[0]:
        st.download_button(
            label="⬇️ Download as CSV",
            data=df.to_csv(index=False),
            file_name=f"processed_documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    with col_dl[1]:
        st.download_button(
            label="⬇️ Download as JSON",
            data=df.to_json(orient="records", indent=2),
            file_name=f"processed_documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )