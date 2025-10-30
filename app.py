"""
SkillGapAI - Complete Resume Skill Analysis System
Main Application - Streamlit UI
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Import backend modules
sys.path.insert(0, str(Path(__file__).parent))
from backend.data_ingestion import DataIngestion
from backend.skill_extraction import SkillExtractor
from backend.gap_analysis import SkillGapAnalyzer
from backend.visualizations import Visualizer
from backend.export_manager import ExportManager
from config import APP_TITLE, APP_ICON, PAGE_LAYOUT, SIMILARITY_THRESHOLD, SKILL_CATEGORIES

# Page Configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

# Enhanced Beautiful Custom CSS
st.markdown("""
    <style>
    /* Ultra Stunning Main Theme */
    .stApp {
        background: linear-gradient(135deg, 
            #667eea 0%, 
            #764ba2 25%,
            #f093fb 50%,
            #4facfe 75%,
            #00f2fe 100%
        );
        background-attachment: fixed;
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Stunning Elegant Header */
    .main-header {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.96) 0%, rgba(118, 75, 162, 0.96) 50%, rgba(240, 147, 251, 0.96) 100%);
        backdrop-filter: blur(15px);
        padding: 4rem 2rem;
        border-radius: 25px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 25px 70px rgba(0,0,0,0.35), 
                    inset 0 1px 0 rgba(255,255,255,0.2);
        border: 2px solid rgba(255,255,255,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3.5rem;
        font-weight: 900;
        text-shadow: 3px 3px 8px rgba(0,0,0,0.3), 
                    0 0 20px rgba(255,255,255,0.2);
        letter-spacing: -2px;
        position: relative;
        z-index: 1;
    }
    
    /* Beautiful metric cards with glass effect */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Ultra Stunning Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 200%;
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 2.5rem;
        font-weight: 800;
        font-size: 1.15rem;
        transition: all 0.4s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5),
                    0 2px 8px rgba(0,0,0,0.2);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255,255,255,0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton>button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.7),
                    0 4px 12px rgba(0,0,0,0.3);
        background-position: 100% 50%;
    }
    
    .stButton>button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    /* Sidebar with gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.98) 0%, rgba(118, 75, 162, 0.98) 100%);
    }
    
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stSlider label {
        color: white !important;
    }
    
    /* Success messages */
    .stSuccess {
        background: rgba(46, 204, 113, 0.1);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Section headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 700;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
    
    /* Add shimmer effect */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .loading {
        background: linear-gradient(
            90deg,
            rgba(255,255,255,0.1) 0%,
            rgba(255,255,255,0.2) 50%,
            rgba(255,255,255,0.1) 100%
        );
        background-size: 1000px 100%;
        animation: shimmer 2s infinite;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.uploaded_files = {}
    st.session_state.parsed_documents = {}
    # Do not pre-create extracted skill or gap keys — create them when results are available.
    # Pre-creating them (even as empty dicts) prevents the extraction/analysis steps from running
    # because the code checks for the presence of these keys in session_state.
    st.session_state.current_step = 0

# Stunning Header with enhanced styling
st.markdown(f"""
    <div class="main-header">
        <h1 style="font-size: 3.5rem; margin-bottom: 1rem;">{APP_ICON} SkillGapAI</h1>
        <p style="font-size: 1.5rem; margin-top: 0; font-weight: 400; letter-spacing: 1px;">
            AI-Powered Resume Skill Analysis System
        </p>
        <div style="font-size:1rem; margin-top:1rem; opacity:0.95;">
            <!-- Dynamic step progress UI will be rendered below by Streamlit code -->
        </div>
    </div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Model Selection
st.sidebar.subheader("Model Settings")
model_selection = st.sidebar.selectbox(
    "Select Embedding Model",
    ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"],
    help="Choose the Sentence-BERT model for embeddings"
)

# Threshold Settings
st.sidebar.subheader("Analysis Thresholds")
similarity_threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(SIMILARITY_THRESHOLD),
    step=0.05,
    help="Minimum similarity score for skill matching"
)

confidence_threshold = st.sidebar.slider(
    "Confidence Filter",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
    help="Minimum confidence for skill extraction"
)

# Action Buttons
st.sidebar.markdown("---")
reset_button = st.sidebar.button("🔄 Reset Analysis", use_container_width=True)
export_button = st.sidebar.button("📥 Export Results", use_container_width=True)

if reset_button:
    st.session_state.clear()
    st.rerun()

# Main App Content
def main():
    """Main application logic"""
    # Render dynamic step indicator based on session state
    def render_step_indicator():
        # Steps: 1 Upload, 2 Extract, 3 Analyze, 4 Gaps, 5 Visualize, 6 Export
        current = st.session_state.get('current_step', 0)
        # Inline CSS for step indicator
        css = f"""
        <style>
        .step-row {{display:flex; gap:2rem; justify-content:center; align-items:center; margin-top:0.5rem;}}
        .step {{text-align:center;}}
        .circle {{width:44px; height:44px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; color:white; font-weight:800; margin:0 auto;}}
        .label {{margin-top:6px; font-size:0.85rem; color:#ffffffcc}}
        .inactive {{background:#b2bec3; opacity:0.9}}
        .active {{background:linear-gradient(135deg,#4facfe,#667eea); box-shadow:0 6px 18px rgba(0,0,0,0.25)}}
        </style>
        """

        # Build HTML for each step, mark active if step <= current
        def step_html(n, label):
            cls = 'active' if current >= n else 'inactive'
            return f"<div class='step'><div class='circle {cls}'>{n}</div><div class='label'>{label}</div></div>"

        html = css + "<div class='step-row'>"
        html += step_html(1, 'Upload')
        html += step_html(2, 'Extract')
        html += step_html(3, 'Analyze')
        html += step_html(4, 'Gaps')
        html += step_html(5, 'Visualize')
        html += step_html(6, 'Export')
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    render_step_indicator()
    
    # Document Upload
    st.header("📄 Upload Documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Resume")
        resume_file = st.file_uploader(
            "Upload Resume",
            type=['pdf', 'docx', 'txt'],
            key="resume_upload",
            help="Upload your resume in PDF, DOCX, or TXT format"
        )
    
    with col2:
        st.subheader("Job Description")
        jd_file = st.file_uploader(
            "Upload Job Description",
            type=['pdf', 'docx', 'txt'],
            key="jd_upload",
            help="Upload the job description in PDF, DOCX, or TXT format"
        )
    
    # Process uploaded files
    if resume_file and jd_file:
        ingestion = DataIngestion()
        
        # File Upload Status Display
        st.subheader("📊 File Upload Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.caption("Resume")
            st.write(f"📄 {resume_file.name}")
            st.write(f"📁 Type: {Path(resume_file.name).suffix}")
            st.write(f"📦 Size: {len(resume_file.getvalue()) / 1024:.1f} KB")
        
        with col2:
            st.caption("Job Description")
            st.write(f"📄 {jd_file.name}")
            st.write(f"📁 Type: {Path(jd_file.name).suffix}")
            st.write(f"📦 Size: {len(jd_file.getvalue()) / 1024:.1f} KB")
        
        # Process Resume
        if resume_file.name not in st.session_state.uploaded_files:
            with st.spinner(f"Processing {resume_file.name}..."):
                progress_bar = st.progress(0)
                st.session_state.uploaded_files[resume_file.name] = {
                    'name': resume_file.name,
                    'type': Path(resume_file.name).suffix,
                    'size': len(resume_file.getvalue()),
                    'status': 'processing'
                }
                progress_bar.progress(50)
                
                # Parse document
                file_info_resume = st.session_state.uploaded_files[resume_file.name]
                raw_text, cleaned_text, error = ingestion.parse_document(resume_file, file_info_resume)
                
                if not error:
                    st.session_state.parsed_documents['resume'] = {
                        'raw': raw_text,
                        'cleaned': cleaned_text,
                        'file_info': file_info_resume
                    }
                    # Mark upload step complete when both resume and jd are parsed
                    st.session_state.current_step = 1
                    progress_bar.progress(100)
                    st.success("✅ Resume processed successfully!")
                    
                    # Display word count and normalization summary
                    st.metric("Word Count", file_info_resume.get('word_count', 0))
                    
                    # Text Preview Toggle (show full raw text in an expander when checked)
                    show_raw = st.checkbox("Show Raw Text", key="resume_raw")
                    if show_raw:
                        # Expand by default when user requests raw text
                        with st.expander("Raw Text Preview", expanded=True):
                            # Show the full extracted raw text (not truncated) so user can inspect it
                            st.text_area("Raw Text", value=raw_text, height=400, key="resume_raw_text")
                else:
                    st.error(f"❌ Error: {error}")
        
        # Process Job Description
        if jd_file.name not in st.session_state.uploaded_files:
            with st.spinner(f"Processing {jd_file.name}..."):
                progress_bar = st.progress(0)
                st.session_state.uploaded_files[jd_file.name] = {
                    'name': jd_file.name,
                    'type': Path(jd_file.name).suffix,
                    'size': len(jd_file.getvalue()),
                    'status': 'processing'
                }
                progress_bar.progress(50)
                
                # Parse document
                file_info_jd = st.session_state.uploaded_files[jd_file.name]
                raw_text, cleaned_text, error = ingestion.parse_document(jd_file, file_info_jd)
                
                if not error:
                    st.session_state.parsed_documents['jd'] = {
                        'raw': raw_text,
                        'cleaned': cleaned_text,
                        'file_info': file_info_jd
                    }
                    # Mark upload step complete when both resume and jd are parsed
                    st.session_state.current_step = 1
                    progress_bar.progress(100)
                    st.success("✅ Job Description processed successfully!")
                    
                    # Display word count
                    st.metric("Word Count", file_info_jd.get('word_count', 0))
                    
                    # Text Preview Toggle (show full raw text in an expander when checked)
                    show_raw = st.checkbox("Show Raw Text", key="jd_raw")
                    if show_raw:
                        with st.expander("Raw Text Preview", expanded=True):
                            st.text_area("Raw Text", value=raw_text, height=400, key="jd_raw_text")
                else:
                    st.error(f"❌ Error: {error}")
        
        # Skill Extraction - Auto-run if documents are parsed
        if 'resume' in st.session_state.parsed_documents and 'jd' in st.session_state.parsed_documents:
            
            # Check if skills already extracted
            needs_extraction = 'extracted_skills_resume' not in st.session_state or 'extracted_skills_jd' not in st.session_state
            
            if needs_extraction:
                st.header("🔍 Extract Skills")
                # Allow user to choose auto-run or manual extraction
                auto_extract = st.checkbox("Auto-run extraction after upload", value=True, help="If enabled, extraction will start automatically once both documents are parsed")
                extract_button = st.button("Extract Skills from Documents", type="primary", use_container_width=True)
                if auto_extract:
                    extract_button = True
            else:
                extract_button = True  # Auto-continue if already extracted

            # Diagnostics: show available NLP/embedding support
            with st.expander("⚙️ Diagnostics (model availability)", expanded=False):
                # Check spaCy/ sentence-transformers availability via lazy import
                try:
                    import importlib
                    spacy_spec = importlib.util.find_spec('spacy')
                    sbert_spec = importlib.util.find_spec('sentence_transformers')
                    st.write('spaCy installed:', bool(spacy_spec))
                    st.write('Sentence-Transformers installed:', bool(sbert_spec))
                    st.write('Configured SBERT model:', model_selection)
                    st.write('Similarity threshold:', similarity_threshold)
                except Exception as e:
                    st.write('Diagnostics error:', str(e))
            
            if extract_button:
                
                # Extract from Resume
                if 'extracted_skills_resume' not in st.session_state:
                    with st.spinner("Extracting skills from Resume..."):
                        try:
                            # Create one extractor instance and reuse
                            if 'extractor_instance' not in st.session_state:
                                st.session_state.extractor_instance = SkillExtractor()
                            extractor = st.session_state.extractor_instance

                            resume_text = st.session_state.parsed_documents['resume']['cleaned']
                            if resume_text and len(resume_text) > 0:
                                st.session_state.extracted_skills_resume = extractor.extract_skills(
                                    resume_text, 
                                    confidence_threshold=confidence_threshold
                                )
                                st.success("✅ Resume skills extracted!")
                                # mark extract step (2) when extraction completes
                                st.session_state.current_step = max(st.session_state.get('current_step', 0), 2)
                            else:
                                st.error("No text found in resume")
                        except Exception as e:
                            st.error(f"Error extracting resume skills: {str(e)}")
                
                # Extract from Job Description
                if 'extracted_skills_jd' not in st.session_state:
                    with st.spinner("Extracting skills from Job Description..."):
                        try:
                            # Reuse extractor instance
                            extractor = st.session_state.get('extractor_instance') or SkillExtractor()
                            jd_text = st.session_state.parsed_documents['jd']['cleaned']
                            if jd_text and len(jd_text) > 0:
                                st.session_state.extracted_skills_jd = extractor.extract_skills(
                                    jd_text, 
                                    confidence_threshold=confidence_threshold
                                )
                                st.success("✅ Job Description skills extracted!")
                                # mark extract step when JD extraction completes
                                st.session_state.current_step = max(st.session_state.get('current_step', 0), 2)
                            else:
                                st.error("No text found in job description")
                        except Exception as e:
                            st.error(f"Error extracting JD skills: {str(e)}")
                
                # Display extracted skills ONLY if both are available
                if 'extracted_skills_resume' in st.session_state and 'extracted_skills_jd' in st.session_state:
                    if st.session_state.extracted_skills_resume and st.session_state.extracted_skills_jd:
                        st.success(f"✅ All Skills Extracted Successfully!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📝 Resume Skills")
                            resume_skills = list(st.session_state.extracted_skills_resume.get('skill_counts', {}).keys())
                            st.metric("Total Skills Found", len(resume_skills))
                            
                            if st.session_state.extracted_skills_resume.get('skill_counts'):
                                # Create tag cloud for top skills
                                skills_df = pd.DataFrame([
                                    {'skill': skill, 'frequency': count}
                                    for skill, count in list(st.session_state.extracted_skills_resume['skill_counts'].items())[:15]
                                ])
                                
                                vis = Visualizer()
                                tag_chart = vis.create_skill_frequency_chart(skills_df)
                                if tag_chart:
                                    st.plotly_chart(tag_chart, use_container_width=True)
                            
                            with st.expander("View All Resume Skills"):
                                st.write(resume_skills[:10] if len(resume_skills) > 10 else resume_skills)

                        # Show raw extracted structure for debugging
                        with st.expander("Raw Extracted Skills (Resume)"):
                            st.write(st.session_state.extracted_skills_resume)
                        
                        with col2:
                            st.subheader("📋 Job Description Skills")
                            jd_skills = list(st.session_state.extracted_skills_jd.get('skill_counts', {}).keys())
                            st.metric("Total Skills Found", len(jd_skills))
                            
                            if st.session_state.extracted_skills_jd.get('skill_counts'):
                                # Create tag cloud for top skills
                                skills_df = pd.DataFrame([
                                    {'skill': skill, 'frequency': count}
                                    for skill, count in list(st.session_state.extracted_skills_jd['skill_counts'].items())[:15]
                                ])
                                
                                vis = Visualizer()
                                tag_chart = vis.create_skill_frequency_chart(skills_df)
                                if tag_chart:
                                    st.plotly_chart(tag_chart, use_container_width=True)
                            
                            with st.expander("View All JD Skills"):
                                st.write(jd_skills[:10] if len(jd_skills) > 10 else jd_skills)

                        with st.expander("Raw Extracted Skills (Job Description)"):
                            st.write(st.session_state.extracted_skills_jd)
                        
                        # Gap Analysis
                        st.header("🎯 Analyze Skill Gap")

                        # Allow auto-run analysis after extraction
                        auto_analyze = st.checkbox("Auto-run gap analysis", value=True, help="If enabled, analysis runs automatically after skill extraction")
                        analyze_button = st.button("Run Gap Analysis", type="primary", use_container_width=True)

                        if auto_analyze:
                            analyze_button = True

                        if analyze_button or 'gap_analysis' in st.session_state:
                            if 'gap_analysis' not in st.session_state:
                                with st.spinner("Analyzing skill gaps..."):
                                    analyzer = SkillGapAnalyzer(model_name=model_selection)

                                    # Get skills as lists
                                    resume_skills_list = list(st.session_state.extracted_skills_resume.get('skill_counts', {}).keys())
                                    jd_skills_list = list(st.session_state.extracted_skills_jd.get('skill_counts', {}).keys())

                                    st.session_state.gap_analysis = analyzer.analyze_gap(
                                        resume_skills_list,
                                        jd_skills_list,
                                        threshold=similarity_threshold
                                    )
                                    # mark analyze step complete
                                    st.session_state.current_step = max(st.session_state.get('current_step', 0), 3)

                            # Show raw gap_analysis for debugging
                            with st.expander("Raw Gap Analysis"):
                                st.write(st.session_state.gap_analysis)
                            
                            # Display Results
                            if st.session_state.gap_analysis:
                                st.header("📊 Analysis Results")
                                
                                # Summary Metrics
                                vis = Visualizer()
                                summary = vis.create_match_summary_card(st.session_state.gap_analysis)
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric(
                                        "Overall Match",
                                        f"{summary['overall_match']:.1f}%"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Total Skills",
                                        summary['total_skills']
                                    )
                                
                                with col3:
                                    st.metric(
                                        "Matched Skills",
                                        summary['matched']
                                    )
                                
                                with col4:
                                    st.metric(
                                        "Missing Skills",
                                        summary['gaps']
                                    )
                                
                                # Visualizations
                                st.subheader("📈 Visualizations")
                                
                                # Radar Chart
                                # Ensure we have an analyzer instance available (may not exist if analysis was run earlier)
                                try:
                                    analyzer = SkillGapAnalyzer(model_name=model_selection)
                                except Exception:
                                    analyzer = None

                                if analyzer is not None:
                                    category_stats = analyzer.get_category_analysis(
                                        st.session_state.gap_analysis,
                                        SKILL_CATEGORIES
                                    )
                                else:
                                    category_stats = {}

                                if category_stats:
                                    radar_chart = vis.create_radar_chart(category_stats)
                                    if radar_chart:
                                        st.plotly_chart(radar_chart, use_container_width=True)
                                
                                # Similarity Heatmap
                                heatmap = vis.create_similarity_heatmap(st.session_state.gap_analysis)
                                if heatmap:
                                    st.plotly_chart(heatmap, use_container_width=True)
                                
                                # Gap Bar Chart
                                gap_chart = vis.create_gap_bar_chart(st.session_state.gap_analysis)
                                if gap_chart:
                                    st.plotly_chart(gap_chart, use_container_width=True)

                                # If any visualization is produced, mark visualize step complete
                                st.session_state.current_step = max(st.session_state.get('current_step', 0), 5)
                                
                                # Top Missing Skills
                                st.subheader("⚠️ Top Missing Skills")
                                gaps = st.session_state.gap_analysis.get('gaps', [])

                                if gaps:
                                    top_gaps = sorted(gaps, key=lambda x: x['similarity'])[:10]
                                    
                                    gap_data = []
                                    for gap in top_gaps:
                                        gap_data.append({
                                            'Skill': gap['jd_skill'],
                                            'Best Match': gap.get('resume_match', 'N/A'),
                                            'Similarity Score': f"{gap['similarity']*100:.1f}%",
                                            'Status': 'Missing'
                                        })
                                    
                                    gap_df = pd.DataFrame(gap_data)
                                    st.dataframe(gap_df, use_container_width=True, hide_index=True)
                                    # Mark gaps step complete
                                    st.session_state.current_step = max(st.session_state.get('current_step', 0), 4)
                                
                                # Detailed Match Table
                                st.subheader("📋 Detailed Match Table")
                                matches = st.session_state.gap_analysis.get('matches', [])
                                
                                if matches:
                                    detailed_data = []
                                    for match in matches:
                                        detailed_data.append({
                                            'JD Skill': match['jd_skill'],
                                            'Resume Match': match['resume_match'],
                                            'Similarity': f"{match['similarity']*100:.1f}%",
                                            'Status': match['match_status'].title()
                                        })
                                    
                                    detailed_df = pd.DataFrame(detailed_data)
                                    st.dataframe(detailed_df, use_container_width=True, hide_index=True)
                                
                                # Export Options
                                st.subheader("📤 Export Results")
                                export_manager = ExportManager()

                                # Always show both CSV and PDF download options when analysis is present
                                csv_path = "skill_gap_analysis.csv"
                                export_manager.export_to_csv(st.session_state.gap_analysis, csv_path)

                                # CSV download button
                                try:
                                    with open(csv_path, "rb") as file:
                                        st.download_button(
                                            label="Download CSV Report",
                                            data=file,
                                            file_name=csv_path,
                                            mime="text/csv",
                                            key="download_csv"
                                        )
                                        # mark export step when user initiates download via sidebar export button
                                        if export_button:
                                            st.session_state.current_step = max(st.session_state.get('current_step', 0), 6)
                                except Exception as e:
                                    st.error(f"CSV export error: {str(e)}")

                                # PDF export (if reportlab available)
                                pdf_path = "skill_gap_analysis.pdf"
                                pdf_msg = export_manager.export_to_pdf(st.session_state.gap_analysis, pdf_path)

                                if export_manager.has_pdf and pdf_msg and pdf_msg.startswith("PDF exported to"):
                                    try:
                                        with open(pdf_path, "rb") as pf:
                                            st.download_button(
                                                label="Download PDF Report",
                                                data=pf,
                                                file_name=pdf_path,
                                                mime="application/pdf",
                                                key="download_pdf"
                                            )
                                    except Exception as e:
                                        st.error(f"PDF download error: {str(e)}")
                                else:
                                    # Inform user PDF not available and provide message from exporter
                                    if not export_manager.has_pdf:
                                        st.info("PDF export not available. Install 'reportlab' to enable PDF reports.")
                                    elif pdf_msg:
                                        st.info(pdf_msg)

if __name__ == "__main__":
    main()

