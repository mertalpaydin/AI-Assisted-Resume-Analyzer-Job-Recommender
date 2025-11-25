"""
AI-Assisted Resume Analyzer & Job Recommender - Streamlit UI
Phase 6: Streamlit Application

A web interface for matching resumes to relevant job postings using
semantic similarity and skill-based analysis.

Author: Mert Alp Aydin
Date: 2025-11-22
"""

import streamlit as st
import logging
from pathlib import Path
import tempfile
import os
import sys
from typing import Optional, Dict, Any
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from matching_engine import MatchingEngine, MatchingResult
from report_generator import ReportGenerator, MatchingReport
from resume_schema import Resume

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================
# PAGE CONFIGURATION
# ====================

st.set_page_config(
    page_title="AI Resume Analyzer & Job Recommender",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================
# CUSTOM CSS
# ====================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .job-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        border-left: 4px solid #3498db;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .skill-tag {
        display: inline-block;
        padding: 5px 12px;
        margin: 3px;
        border-radius: 15px;
        font-size: 0.85em;
        font-weight: 500;
    }
    .skill-matched {
        background: #d4edda;
        color: #155724;
    }
    .skill-missing {
        background: #f8d7da;
        color: #721c24;
    }
    .skill-extra {
        background: #d1ecf1;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# ====================
# SESSION STATE INITIALIZATION
# ====================

if 'matching_result' not in st.session_state:
    st.session_state.matching_result = None
if 'matching_report' not in st.session_state:
    st.session_state.matching_report = None
if 'engine' not in st.session_state:
    st.session_state.engine = None


# ====================
# HELPER FUNCTIONS
# ====================

@st.cache_resource
def load_matching_engine(embeddings_path: str = "data/embeddings"):
    """Load and cache the matching engine."""
    logger.info("Loading matching engine...")
    try:
        engine = MatchingEngine(embeddings_path=embeddings_path)
        return engine, None
    except Exception as e:
        logger.error(f"Failed to load matching engine: {e}")
        return None, str(e)


def process_resume(
    uploaded_file,
    engine: MatchingEngine,
    top_k: int = 10,
    use_mmr: bool = True
) -> tuple[Optional[MatchingResult], Optional[str]]:
    """Process uploaded resume and find matches."""
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = Path(tmp_file.name)

        # Update engine settings
        engine.use_mmr = use_mmr

        # Run matching
        result = engine.match_resume_pdf(tmp_path, top_k=top_k)

        # Clean up temp file
        os.unlink(tmp_path)

        return result, None
    except Exception as e:
        logger.error(f"Error processing resume: {e}")
        return None, str(e)


def display_resume_summary(resume: Resume):
    """Display resume summary in a nice format."""
    st.subheader("📋 Resume Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Name:**")
        st.write(resume.full_name)
        if resume.email:
            st.markdown("**Email:**")
            st.write(resume.email)

    with col2:
        if resume.phone:
            st.markdown("**Phone:**")
            st.write(resume.phone)
        if resume.location:
            st.markdown("**Location:**")
            st.write(resume.location)

    with col3:
        st.markdown("**Skills Count:**")
        st.write(len(resume.skills))
        st.markdown("**Experience:**")
        st.write(f"{len(resume.experience)} positions")

    # Skills
    if resume.skills:
        st.markdown("**Top Skills:**")
        skills_html = " ".join([f'<span class="skill-tag skill-extra">{s}</span>'
                               for s in resume.skills[:15]])
        st.markdown(skills_html, unsafe_allow_html=True)


def display_job_match(match, rank: int):
    """Display a single job match card."""
    job = match.job

    # Card container
    with st.container():
        # Header row
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {rank}. {job.title}")
            st.markdown(f"**{job.company}** • {job.location or 'Location not specified'}")
        with col2:
            # Similarity score
            score_pct = match.similarity_score * 100
            if score_pct >= 70:
                color = "🟢"
            elif score_pct >= 50:
                color = "🟡"
            else:
                color = "🔴"
            st.metric("Similarity", f"{score_pct:.1f}%", delta=color)

        # Skill match info
        if match.skill_match:
            skill_data = match.skill_match

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Skill Match", f"{skill_data['match_percentage']:.0f}%")
            with col2:
                st.metric("Matched Skills", len(skill_data['matched_skills']))
            with col3:
                st.metric("Missing Skills", len(skill_data['missing_skills']))

            # Skills breakdown
            with st.expander("📊 Skills Breakdown"):
                # Matched skills
                if skill_data['matched_skills']:
                    st.markdown("**✅ Matched Skills:**")
                    matched_html = " ".join([f'<span class="skill-tag skill-matched">{s}</span>'
                                            for s in skill_data['matched_skills'][:10]])
                    st.markdown(matched_html, unsafe_allow_html=True)

                # Missing skills
                if skill_data['missing_skills']:
                    st.markdown("**❌ Missing Skills:**")
                    missing_html = " ".join([f'<span class="skill-tag skill-missing">{s}</span>'
                                            for s in skill_data['missing_skills'][:10]])
                    st.markdown(missing_html, unsafe_allow_html=True)

        # Job description (truncated)
        with st.expander("📝 Job Description"):
            st.write(job.description[:500] + "..." if len(job.description) > 500 else job.description)

        st.markdown("---")


def create_skill_analysis_chart(report: MatchingReport):
    """Create skill analysis visualization."""
    # Extract skill match percentages
    job_titles = [rec.job_title[:30] + "..." if len(rec.job_title) > 30 else rec.job_title
                  for rec in report.recommendations]
    skill_matches = [rec.skill_match_percentage for rec in report.recommendations]
    similarities = [rec.similarity_score * 100 for rec in report.recommendations]

    # Create grouped bar chart
    fig = go.Figure(data=[
        go.Bar(name='Skill Match %', x=job_titles, y=skill_matches, marker_color='#3498db'),
        go.Bar(name='Similarity %', x=job_titles, y=similarities, marker_color='#9b59b6')
    ])

    fig.update_layout(
        title="Skill Match vs. Similarity Score by Job",
        xaxis_title="Job Position",
        yaxis_title="Percentage",
        barmode='group',
        height=400,
        showlegend=True
    )

    return fig


def create_skill_demand_chart(report: MatchingReport):
    """Create chart showing most in-demand skills."""
    skill_profile = report.skill_profile_summary

    if 'skill_demand_breakdown' in skill_profile:
        skills = list(skill_profile['skill_demand_breakdown'].keys())[:10]
        counts = list(skill_profile['skill_demand_breakdown'].values())[:10]

        # Create DataFrame for plotly
        df = pd.DataFrame({
            'Skill': skills,
            'Count': counts
        })

        fig = px.bar(
            df,
            x='Count',
            y='Skill',
            orientation='h',
            title="Most In-Demand Skills (Across Top Matches)",
            labels={'Count': 'Number of Jobs', 'Skill': 'Skill'},
            color='Count',
            color_continuous_scale='Blues'
        )

        fig.update_layout(height=400, showlegend=False)
        return fig

    return None


# ====================
# MAIN APPLICATION
# ====================

def main():
    # Header
    st.markdown('<h1 class="main-header">📄 AI Resume Analyzer & Job Recommender</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Match your resume to relevant job postings using AI-powered semantic analysis</p>',
                unsafe_allow_html=True)

    # ====================
    # SIDEBAR - CONFIGURATION
    # ====================

    with st.sidebar:
        st.header("⚙️ Configuration")

        # Number of results
        top_k = st.slider(
            "Number of job matches",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            help="How many top matching jobs to display"
        )

        # MMR toggle
        use_mmr = st.toggle(
            "Use MMR for diversity",
            value=True,
            help="Maximum Marginal Relevance balances similarity with diversity"
        )

        st.markdown("---")

        # About section
        st.header("ℹ️ About")
        st.markdown("""
        This application uses:
        - **EmbeddingGemma** for embeddings
        - **FAISS** for vector search
        - **MMR** for diverse results
        - **gemma3:4b** for resume parsing
        - **RAKE + granite4:micro** for skill extraction
        """)

    # ====================
    # MAIN AREA - FILE UPLOAD
    # ====================

    # Load matching engine
    if st.session_state.engine is None:
        with st.spinner("🔄 Loading matching engine..."):
            engine, error = load_matching_engine()
            if error:
                st.error(f"❌ Failed to load matching engine: {error}")
                st.info("💡 Make sure you've run `python src/generate_job_embeddings.py` first!")
                st.stop()
            st.session_state.engine = engine
            st.success("✅ Matching engine loaded successfully!")

    engine = st.session_state.engine

    # File upload
    st.subheader("1️⃣ Upload Your Resume")
    uploaded_file = st.file_uploader(
        "Choose a PDF resume",
        type=['pdf'],
        help="Upload your resume in PDF format"
    )

    # Process button
    if uploaded_file is not None:
        st.success(f"📄 File uploaded: {uploaded_file.name}")

        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            process_button = st.button("🚀 Find Matching Jobs", type="primary", use_container_width=True)
        with col2:
            if st.session_state.matching_result:
                clear_button = st.button("🔄 Clear Results", use_container_width=True)
                if clear_button:
                    st.session_state.matching_result = None
                    st.session_state.matching_report = None
                    st.rerun()

        if process_button:
            # Process resume
            with st.spinner("🔄 Processing resume and finding matches..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Step 1: Parse resume
                status_text.text("📄 Parsing resume...")
                progress_bar.progress(25)
                time.sleep(0.5)

                # Step 2: Generate embeddings
                status_text.text("🧠 Generating embeddings...")
                progress_bar.progress(50)

                # Step 3: Search
                status_text.text("🔍 Searching for matching jobs...")
                progress_bar.progress(75)

                # Process
                result, error = process_resume(uploaded_file, engine, top_k, use_mmr)

                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()

                if error:
                    st.error(f"❌ Error: {error}")
                else:
                    st.session_state.matching_result = result

                    # Generate report
                    report_gen = ReportGenerator()
                    st.session_state.matching_report = report_gen.generate_report(result)

                    st.success(f"✅ Found {len(result.matches)} matching jobs in {result.search_time_seconds:.2f}s!")

    # ====================
    # RESULTS DISPLAY
    # ====================

    if st.session_state.matching_result:
        result: MatchingResult = st.session_state.matching_result
        report: MatchingReport = st.session_state.matching_report

        st.markdown("---")
        st.subheader("2️⃣ Results")

        # Display resume summary
        display_resume_summary(result.resume)

        st.markdown("---")

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview",
            "💼 Job Matches",
            "🎯 Skills Analysis",
            "📥 Export"
        ])

        # TAB 1: OVERVIEW
        with tab1:
            st.header("Match Overview")

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Matches", len(result.matches))
            with col2:
                avg_similarity = sum(m.similarity_score for m in result.matches) / len(result.matches) * 100
                st.metric("Avg Similarity", f"{avg_similarity:.1f}%")
            with col3:
                if result.matches[0].skill_match:
                    avg_skill = sum(m.skill_match['match_percentage'] for m in result.matches
                                  if m.skill_match) / len(result.matches)
                    st.metric("Avg Skill Match", f"{avg_skill:.1f}%")
            with col4:
                st.metric("Search Time", f"{result.search_time_seconds:.2f}s")

            # Skill analysis chart
            st.markdown("### 📊 Match Quality by Job")
            chart = create_skill_analysis_chart(report)
            st.plotly_chart(chart, use_container_width=True)

            # Skill demand chart
            demand_chart = create_skill_demand_chart(report)
            if demand_chart:
                st.markdown("### 🎯 Most Valuable Skills")
                st.plotly_chart(demand_chart, use_container_width=True)

        # TAB 2: JOB MATCHES
        with tab2:
            st.header("Top Matching Jobs")

            # Filter options
            col1, col2 = st.columns([2, 1])
            with col1:
                search_filter = st.text_input("🔍 Filter by job title or company", "")
            with col2:
                min_similarity = st.slider("Min similarity %", 0, 100, 0, 5)

            # Display filtered matches
            filtered_matches = [
                m for m in result.matches
                if (search_filter.lower() in m.job.title.lower() or
                    search_filter.lower() in m.job.company.lower())
                and (m.similarity_score * 100 >= min_similarity)
            ]

            st.write(f"Showing {len(filtered_matches)} of {len(result.matches)} jobs")

            for match in filtered_matches:
                display_job_match(match, match.rank)

        # TAB 3: SKILLS ANALYSIS
        with tab3:
            st.header("Skills Analysis")

            # Development recommendations
            if report.development_recommendations:
                st.subheader("💡 Skill Development Recommendations")
                for rec in report.development_recommendations:
                    st.info(rec)

            # Skill profile
            st.subheader("📈 Your Skill Profile")
            skill_profile = report.skill_profile_summary

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Skills", skill_profile.get('total_skills', 0))
            with col2:
                if 'most_valuable_skills' in skill_profile:
                    st.write("**Most Valuable Skills:**")
                    valuable_html = " ".join([f'<span class="skill-tag skill-matched">{s}</span>'
                                             for s in skill_profile['most_valuable_skills']])
                    st.markdown(valuable_html, unsafe_allow_html=True)

            # All candidate skills
            st.subheader("📋 All Your Skills")
            if report.candidate_skills:
                all_skills_html = " ".join([f'<span class="skill-tag skill-extra">{s}</span>'
                                           for s in report.candidate_skills])
                st.markdown(all_skills_html, unsafe_allow_html=True)

        # TAB 4: EXPORT
        with tab4:
            st.header("Export Results")

            st.write("Download your matching report in different formats:")

            report_gen = ReportGenerator()

            col1, col2, col3 = st.columns(3)

            with col1:
                # JSON export
                json_data = report_gen.to_json(report)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_data,
                    file_name=f"job_matches_{result.resume.full_name.replace(' ', '_')}.json",
                    mime="application/json",
                    use_container_width=True
                )

            with col2:
                # Markdown export
                md_data = report_gen.to_markdown(report)
                st.download_button(
                    label="📝 Download Markdown",
                    data=md_data,
                    file_name=f"job_matches_{result.resume.full_name.replace(' ', '_')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

            with col3:
                # HTML export
                html_data = report_gen.to_html(report)
                st.download_button(
                    label="🌐 Download HTML",
                    data=html_data,
                    file_name=f"job_matches_{result.resume.full_name.replace(' ', '_')}.html",
                    mime="text/html",
                    use_container_width=True
                )

            # Preview
            st.subheader("📋 Report Preview")
            preview_format = st.radio("Select format to preview:", ["Markdown", "HTML"], horizontal=True)

            if preview_format == "Markdown":
                st.markdown(md_data)
            else:
                st.components.v1.html(html_data, height=800, scrolling=True)

    else:
        # No results yet
        st.info("👆 Upload a resume to get started!")


if __name__ == "__main__":
    main()