"""
Visualization & Dashboard
Creates all visualizations for the analysis results
"""

try:
    import streamlit as st
except Exception:
    class _StFallback:
        pass
    st = _StFallback()

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
except Exception:
    go = None
    px = None
from typing import Dict, List
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # For type hints only (avoid runtime import errors)
    import pandas as pd
    import plotly.graph_objects as _go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SKILL_CATEGORIES


class Visualizer:
    """Creates visualizations for skill analysis and gap reports"""
    
    def __init__(self):
        pass
    
    def create_skill_tag_cloud(self, skills_data: Dict) -> 'go.Figure':
        """Create a tag cloud visualization of top skills"""
        skill_counts = skills_data.get('skill_counts', {})
        
        if not skill_counts:
            return None
        
        # Prepare data (top 20 skills)
        skills_sorted = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        skills = [s[0] for s in skills_sorted]
        counts = [s[1] for s in skills_sorted]
        
        # Create treemap
        fig = go.Figure(go.Treemap(
            labels=skills,
            values=counts,
            parents=['Top Skills'] * len(skills),
            textinfo="label+value"
        ))
        
        fig.update_layout(
            title="Top Skills Tag Cloud",
            height=400
        )
        
        return fig
    
    def create_radar_chart(self, category_stats: Dict) -> 'go.Figure':
        """Create radar chart comparing categories"""
        if not category_stats:
            return None
        
        categories = []
        match_scores = []
        
        for category, stats in category_stats.items():
            if stats['total'] > 0:
                categories.append(category)
                match_scores.append(stats.get('match_percentage', 0))
        
        if not categories:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=match_scores,
            theta=categories,
            fill='toself',
            name='Match Percentage'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Skill Match by Category (Radar Chart)",
            height=400
        )
        
        return fig
    
    def create_similarity_heatmap(self, gap_analysis: Dict) -> 'go.Figure':
        """Create heatmap showing similarity matrix"""
        if 'similarity_matrix' not in gap_analysis:
            return None
        
        jd_skills = gap_analysis.get('jd_skills', [])
        resume_skills = gap_analysis.get('resume_skills', [])
        similarity_matrix = np.array(gap_analysis['similarity_matrix'])
        
        if similarity_matrix.size == 0:
            return None
        
        # Limit to top N for readability
        max_skills = 20
        if len(jd_skills) > max_skills:
            jd_skills = jd_skills[:max_skills]
            similarity_matrix = similarity_matrix[:max_skills]
        
        if len(resume_skills) > max_skills:
            resume_skills = resume_skills[:max_skills]
            similarity_matrix = similarity_matrix[:, :max_skills]
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=resume_skills,
            y=jd_skills,
            colorscale='RdYlGn',
            colorbar=dict(title="Similarity Score"),
            hovertemplate='JD Skill: %{y}<br>Resume Skill: %{x}<br>Similarity: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Skill Similarity Heatmap",
            xaxis_title="Resume Skills",
            yaxis_title="Job Description Skills",
            height=600
        )
        
        return fig
    
    def create_gap_bar_chart(self, gap_analysis: Dict) -> 'go.Figure':
        """Create bar chart of top skill gaps"""
        gaps = gap_analysis.get('gaps', [])
        
        if not gaps:
            return None
        
        # Get top 10 gaps
        top_gaps = sorted(gaps, key=lambda x: x['similarity'])[:10]
        
        skills = [g['jd_skill'] for g in top_gaps]
        scores = [g['similarity'] * 100 for g in top_gaps]
        
        fig = go.Figure(data=[
            go.Bar(
                x=skills,
                y=scores,
                marker=dict(
                    color=scores,
                    colorscale='Reds',
                    showscale=True
                ),
                text=[f"{s:.1f}%" for s in scores],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Top Skill Gaps (Lowest Similarity Scores)",
            xaxis_title="Skill",
            yaxis_title="Similarity Score (%)",
            height=400
        )
        
        return fig
    
    def create_match_summary_card(self, gap_analysis: Dict) -> Dict:
        """Create summary statistics"""
        overall_match = gap_analysis.get('overall_match', 0)
        total_jd_skills = gap_analysis.get('total_jd_skills', 0)
        matched_skills = gap_analysis.get('matched_skills', 0)
        gaps_count = len(gap_analysis.get('gaps', []))
        high_priority_gaps = gap_analysis.get('high_priority_gaps', [])
        
        return {
            'overall_match': overall_match,
            'total_skills': total_jd_skills,
            'matched': matched_skills,
            'gaps': gaps_count,
            'high_priority': len(high_priority_gaps)
        }
    
    def create_category_comparison_chart(self, category_stats: Dict) -> 'go.Figure':
        """Create bar chart comparing match percentage across categories"""
        if not category_stats:
            return None
        
        categories = []
        match_percentages = []
        
        for category, stats in category_stats.items():
            categories.append(category)
            match_percentages.append(stats.get('match_percentage', 0))
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=match_percentages,
                marker=dict(
                    color=match_percentages,
                    colorscale='Greens',
                    showscale=True
                ),
                text=[f"{p:.1f}%" for p in match_percentages],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Match Percentage by Category",
            xaxis_title="Category",
            yaxis_title="Match %",
            height=400
        )
        
        return fig
    
    def create_skill_frequency_chart(self, skills_df: 'pd.DataFrame') -> 'go.Figure':
        """Create bar chart of skill frequencies"""
        if skills_df.empty:
            return None
        
        top_n = skills_df.head(15)
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_n['frequency'],
                y=top_n['skill'],
                orientation='h',
                marker=dict(
                    color=top_n['frequency'],
                    colorscale='Blues',
                    showscale=True
                )
            )
        ])
        
        fig.update_layout(
            title="Top Skills by Frequency",
            xaxis_title="Frequency",
            yaxis_title="Skill",
            height=400
        )
        
        return fig


