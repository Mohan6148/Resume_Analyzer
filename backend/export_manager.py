"""
Export Manager - Handles PDF and CSV export of analysis reports
"""

import pandas as pd
from typing import Dict, List
from datetime import datetime
import sys
from pathlib import Path

# Try to import PDF libraries
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class ExportManager:
    """Handles export of analysis results to PDF and CSV"""
    
    def __init__(self):
        self.has_pdf = PDF_AVAILABLE
    
    def export_to_csv(self, gap_analysis: Dict, output_path: str) -> str:
        """Export gap analysis to CSV"""
        matches = gap_analysis.get('matches', [])
        
        if not matches:
            return "No data to export"
        
        df = pd.DataFrame(matches)
        
        # Reorder columns
        df = df[['jd_skill', 'resume_match', 'similarity', 'match_status']]
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        return f"CSV exported to {output_path}"
    
    def export_to_pdf(self, analysis_data: Dict, output_path: str) -> str:
        """Export complete analysis to PDF"""
        if not self.has_pdf:
            return "PDF export not available. Install: pip install reportlab"
        
        try:
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            story = []
            
            styles = getSampleStyleSheet()
            title_style = styles['Heading1']
            heading_style = styles['Heading2']
            
            # Title
            story.append(Paragraph("Skill Gap Analysis Report", title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Summary
            story.append(Paragraph("Summary", heading_style))
            summary = f"Overall Match: {analysis_data.get('overall_match', 0):.1f}%"
            story.append(Paragraph(summary, styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            
            # Build PDF
            doc.build(story)
            
            return f"PDF exported to {output_path}"
            
        except Exception as e:
            return f"PDF export error: {str(e)}"
    
    def prepare_data_for_export(self, gap_analysis: Dict) -> Dict:
        """Prepare analysis data for export"""
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_match': gap_analysis.get('overall_match', 0),
            'total_skills': gap_analysis.get('total_jd_skills', 0),
            'matched_skills': gap_analysis.get('matched_skills', 0),
            'gaps': gap_analysis.get('gaps', []),
            'matches': gap_analysis.get('matches', [])
        }

