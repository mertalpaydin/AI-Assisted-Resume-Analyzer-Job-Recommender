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
if 'cached_top_k' not in st.session_state:
    st.session_state.cached_top_k = 0  # Track how many jobs were extracted
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None  # Track current file


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


def parse_location(location):
    """Parse location from dict or string format."""
    if isinstance(location, dict):
        # Priority 1: Use formatted if available and non-empty
        if location.get('formatted') and location['formatted'].strip():
            return location['formatted'].strip()

        # Priority 2: Use addressLine if available (often contains "Remote" or full address)
        if location.get('addressLine') and location['addressLine'].strip():
            addr = location['addressLine'].strip()
            if addr.lower() not in ['', 'null', 'none']:
                return addr

        # Priority 3: Build from city, state, country
        parts = []

        # Check if city is "Remote" (special case)
        city = location.get('city', '').strip()
        if city and city.lower() == 'remote':
            return 'Remote'

        # Build location from components
        if city:
            parts.append(city)

        state = location.get('state', '').strip()
        if state:
            parts.append(state.upper() if len(state) == 2 else state)

        # Only add country if it's not US (assume US is default)
        country = location.get('country', '').strip()
        if country and country.upper() != 'US':
            parts.append(country.upper())

        if parts:
            return ', '.join(parts)

        # Last resort: check for any non-empty field
        for key in ['district', 'county', 'quarter']:
            val = location.get(key, '').strip()
            if val:
                return val

        return 'Location not specified'

    elif isinstance(location, str):
        return location.strip() if location and location.strip() else 'Location not specified'

    return 'Location not specified'


def display_job_match(match, rank: int):
    """Display a single job match card."""
    job = match.job

    # Card container
    with st.container():
        # Header row
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {rank}. {job.title}")
            # Parse location properly
            location_str = parse_location(job.location)
            st.markdown(f"**{job.company}** • {location_str}")
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

        # Job description (full)
        with st.expander("📝 Job Description"):
            st.write(job.description)

        # AI Insights Placeholder (Phase 7)
        with st.expander("✨ AI Match Insights (Coming in Phase 7)", expanded=False):
            st.info("🚀 **Coming Soon!**\n\nAI-generated insights will provide:\n- Overall match quality assessment\n- Key strengths for this role\n- Potential concerns or gaps\n- Interview talking points\n\n*This feature will be available in Phase 7*")

        st.markdown("---")


def create_skill_analysis_chart(report: MatchingReport):
    """Create scatter plot showing relationship between similarity and skill match."""
    # Extract metrics
    job_titles = [rec.job_title[:40] + "..." if len(rec.job_title) > 40 else rec.job_title
                  for rec in report.recommendations]
    skill_matches = [rec.skill_match_percentage for rec in report.recommendations]
    similarities = [rec.similarity_score * 100 for rec in report.recommendations]
    companies = [rec.company[:30] for rec in report.recommendations]

    # Create DataFrame
    df = pd.DataFrame({
        'Job Title': job_titles,
        'Company': companies,
        'Similarity %': similarities,
        'Skill Match %': skill_matches,
        'Label': [f"{title}<br>{company}" for title, company in zip(job_titles, companies)]
    })

    # Create scatter plot
    fig = px.scatter(
        df,
        x='Similarity %',
        y='Skill Match %',
        hover_data=['Job Title', 'Company'],
        size=[20] * len(df),  # Fixed size
        color='Skill Match %',
        color_continuous_scale='Viridis',
        title="Match Quality: Similarity vs. Skill Match"
    )

    # Add quadrant lines (average)
    avg_sim = sum(similarities) / len(similarities)
    avg_skill = sum(skill_matches) / len(skill_matches)

    fig.add_hline(y=avg_skill, line_dash="dot", line_color="gray", opacity=0.3,
                  annotation_text=f"Avg Skill: {avg_skill:.1f}%", annotation_position="right")
    fig.add_vline(x=avg_sim, line_dash="dot", line_color="gray", opacity=0.3,
                  annotation_text=f"Avg Similarity: {avg_sim:.1f}%", annotation_position="top")

    fig.update_layout(
        height=500,
        xaxis_title="Semantic Similarity %",
        yaxis_title="Skill Match %",
        showlegend=False
        # Auto-scale axes (no fixed range)
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
            value=5,
            step=1,
            help="How many top matching jobs to display"
        )

        st.markdown("---")

        # About section
        st.header("ℹ️ About")
        st.markdown("""
        This application uses:
        - **EmbeddingGemma** for embeddings
        - **FAISS** for vector search
        - **gemma3:4b** for resume & job skill extraction
        - **Semantic similarity** for skill matching
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

                # Process with default top_k=5
                result, error = process_resume(uploaded_file, engine, 5, use_mmr=True)

                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()

                if error:
                    st.error(f"❌ Error: {error}")
                else:
                    st.session_state.matching_result = result
                    st.session_state.cached_top_k = 5  # Track cached results
                    st.session_state.uploaded_file_name = uploaded_file.name  # Track file

                    # Generate report
                    report_gen = ReportGenerator()
                    st.session_state.matching_report = report_gen.generate_report(result)

                    st.success(f"✅ Found {len(result.matches)} matching jobs in {result.search_time_seconds:.2f}s!")
                    st.rerun()

    # ====================
    # RESULTS DISPLAY
    # ====================

    if st.session_state.matching_result:
        result: MatchingResult = st.session_state.matching_result
        report: MatchingReport = st.session_state.matching_report

        # Check if we need to fetch more results (smart caching)
        if top_k > st.session_state.cached_top_k and uploaded_file is not None:
            # Only re-extract if same file
            if uploaded_file.name == st.session_state.uploaded_file_name:
                with st.spinner(f"🔄 Fetching {top_k} matches..."):
                    # Re-extract with higher top_k
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp_file.name)

                    engine.use_mmr = True
                    new_result = engine.match_resume_pdf(tmp_path, top_k=top_k)
                    os.unlink(tmp_path)

                    st.session_state.matching_result = new_result
                    st.session_state.cached_top_k = top_k

                    # Regenerate report
                    report_gen = ReportGenerator()
                    st.session_state.matching_report = report_gen.generate_report(new_result)

                    result = new_result
                    report = st.session_state.matching_report
                    st.rerun()

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

            # Use top_k for display
            display_matches = result.matches[:top_k]

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Displaying", f"{len(display_matches)}/{len(result.matches)}")
            with col2:
                avg_similarity = sum(m.similarity_score for m in display_matches) / len(display_matches) * 100
                st.metric("Avg Similarity", f"{avg_similarity:.1f}%")
            with col3:
                if display_matches[0].skill_match:
                    avg_skill = sum(m.skill_match['match_percentage'] for m in display_matches
                                  if m.skill_match) / len(display_matches)
                    st.metric("Avg Skill Match", f"{avg_skill:.1f}%")
            with col4:
                st.metric("Search Time", f"{result.search_time_seconds:.2f}s")

            # Skill analysis chart (use top_k only)
            st.markdown("### 📊 Match Quality by Job")
            # Create a filtered report for visualization
            filtered_report_for_chart = MatchingReport(
                candidate_name=report.candidate_name,
                candidate_email=report.candidate_email,
                candidate_skills=report.candidate_skills,
                report_date=report.report_date,
                total_jobs_searched=report.total_jobs_searched,
                search_time_seconds=report.search_time_seconds,
                recommendations=report.recommendations[:top_k],
                skill_profile_summary=report.skill_profile_summary,
                development_recommendations=report.development_recommendations
            )
            chart = create_skill_analysis_chart(filtered_report_for_chart)
            st.plotly_chart(chart, use_container_width=True)

        # TAB 2: JOB MATCHES
        with tab2:
            st.header("Top Matching Jobs")

            # Filter options
            search_filter = st.text_input("🔍 Filter by job title or company", "")

            # Use slider value to limit displayed jobs
            top_k_matches = result.matches[:top_k]
            filtered_matches = [
                m for m in top_k_matches
                if (not search_filter or
                    search_filter.lower() in m.job.title.lower() or
                    search_filter.lower() in m.job.company.lower())
            ]

            st.write(f"Showing {len(filtered_matches)} of {top_k} jobs")

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