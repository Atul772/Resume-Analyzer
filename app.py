# app.py - Enhanced Smart Career Coach

import streamlit as st
from resume_parser import ResumeParser
from pinecone_handler import PineconeHandler
from gemini_api import GeminiAPI
import tempfile
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import re

st.set_page_config(
    page_title="Smart Career Coach", 
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        transition: transform 0.2s;
    }
    .feature-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s;
    }
    .metric-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .contact-chip {
        display: inline-block;
        background: #e3f2fd;
        color: #1976d2;
        padding: 6px 16px;
        margin: 4px;
        border-radius: 20px;
        font-size: 14px;
    }
    .skill-tag {
        display: inline-block;
        background: #e8f5e9;
        color: #2e7d32;
        padding: 4px 12px;
        margin: 4px;
        border-radius: 16px;
        font-size: 13px;
        font-weight: 500;
    }
    .section-header {
        color: #333;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def load_components():
    return PineconeHandler(), GeminiAPI()

# File type validation
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Main app
def main():
    """Main entry point for the Streamlit app."""
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Smart Career Coach</h1>
        <p>AI-Powered Resume Analyzer & Job Role Recommender</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("🚀 Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode:",
        ["📄 Resume Analysis", "🔍 Job Matching", "🤖 AI Career Coach", "📊 Dashboard", "💾 Export Resume"]
    )
    
    # Load components
    try:
        pinecone_handler, gemini_api = load_components()
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        st.info("Please check your API keys in .env file")
        return
    
    # Enhanced file upload with multiple formats
    uploaded_file = st.file_uploader(
        "📤 Upload your resume",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT"
    )
    
    if uploaded_file is not None:
        # Show file info
        file_details = {
            "Filename": uploaded_file.name,
            "FileType": uploaded_file.type,
            "FileSize": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.sidebar.write("📁 **File Details:**")
        for key, value in file_details.items():
            st.sidebar.write(f"- {key}: {value}")

        try:
            with st.spinner("🔍 Analyzing your resume..."):
                parser = ResumeParser()
                
                # Extract text based on file type using the uploaded file object directly
                text = ""
                if uploaded_file.type == "application/pdf":
                    text = parser.extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = parser.extract_text_from_docx(uploaded_file)
                elif uploaded_file.type == "text/plain":
                    text = uploaded_file.read().decode("utf-8")

                if text:
                    # Use the new parsing logic
                    contact_info = parser.extract_contact_info(text)
                    sections = parser.identify_sections(text)
                    
                    # Extract skills separately for other app features
                    if 'skills' in sections:
                        skills_list = parser.extract_skills(sections['skills'])
                    else:
                        skills_list = parser.extract_skills(text) # Fallback to whole text

                    # Combine all parsed data
                    parsed_info = sections
                    parsed_info['contact'] = contact_info
                    parsed_info['skills'] = skills_list
                    
                    # Store in session state
                    st.session_state.parsed_info = parsed_info
                    st.session_state.raw_text = text
                    st.session_state.filename = uploaded_file.name
                else:
                    st.error("Failed to extract text from the resume.")
                    return

        except Exception as e:
            st.error(f"Error processing resume: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return
    
    # Check if we have parsed data
    if 'parsed_info' not in st.session_state:
        show_landing_page()
        return
    
    parsed_info = st.session_state.parsed_info
    raw_text = st.session_state.raw_text
    
    # Route to different modes
    if app_mode == "📄 Resume Analysis":
        show_resume_analysis(parsed_info, raw_text)
    elif app_mode == "🔍 Job Matching":
        show_job_matching(parsed_info, raw_text, pinecone_handler)
    elif app_mode == "🤖 AI Career Coach":
        show_ai_coach(parsed_info, gemini_api)
    elif app_mode == "📊 Dashboard":
        show_dashboard(parsed_info, raw_text)
    elif app_mode == "💾 Export Resume":
        show_export_options(parsed_info, gemini_api)

def show_landing_page():
    """Enhanced landing page when no resume is uploaded."""
    # Hero section
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h2>Welcome to Your AI-Powered Career Assistant! 🚀</h2>
        <p style='font-size: 18px; color: #666;'>
            Upload your resume to unlock personalized career insights and opportunities
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features grid
    col1, col2, col3, col4 = st.columns(4)
    
    features = [
        ("📄", "Resume Analysis", "Extract and analyze all sections of your resume with advanced NLP"),
        ("🔍", "Job Matching", "Find relevant opportunities using semantic search technology"),
        ("🤖", "AI Coaching", "Get personalized feedback from Google's Gemini AI"),
        ("📊", "Analytics", "Visualize your career profile with interactive dashboards")
    ]
    
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"""
            <div class="feature-box" style="text-align: center;">
                <h1>{icon}</h1>
                <h4>{title}</h4>
                <p style="font-size: 14px; color: #666;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # How it works
    st.markdown("### 🎯 How It Works")
    steps = [
        "Upload your resume in PDF, DOCX, or TXT format",
        "AI extracts and analyzes all information",
        "Get matched with relevant job opportunities",
        "Receive personalized improvement suggestions",
        "Export your enhanced resume"
    ]
    
    cols = st.columns(5)
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem;'>
                <div style='background: #667eea; color: white; width: 40px; height: 40px; 
                border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                margin: 0 auto 10px; font-weight: bold;'>{i+1}</div>
                <p style='font-size: 14px;'>{step}</p>
            </div>
            """, unsafe_allow_html=True)

def show_resume_analysis(parsed_info, raw_text):
    """Enhanced resume analysis with contact info and new sections."""
    st.header("📄 Resume Analysis Results")
    
    # Contact Information
    if parsed_info.get("contact"):
        st.markdown("### 👤 Contact Information")
        contact_html = "<div style='background: #f8f9fa; padding: 1rem; border-radius: 8px;'>"
        for key, value in parsed_info["contact"].items():
            if value:
                icon = {
                    "name": "👤",
                    "email": "📧",
                    "phone": "📱",
                    "linkedin": "💼",
                    "github": "💻"
                }.get(key, "📌")
                contact_html += f'<span class="contact-chip">{icon} {value}</span>'
        contact_html += "</div>"
        st.markdown(contact_html, unsafe_allow_html=True)
        st.markdown("---")
    
    # Simplified Metrics overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🧠 Skills Found", len(parsed_info.get("skills", [])))
    with col2:
        sections_found = len([k for k, v in parsed_info.items() if v and k not in ['contact', 'skills', 'raw_text']])
        st.metric("📄 Sections Found", sections_found)
    with col3:
        st.metric("📝 Word Count", len(raw_text.split()))
    
    # Summary Section
    if "summary" in parsed_info and parsed_info["summary"]:
        with st.expander("📝 Professional Summary", expanded=True):
            st.write(parsed_info["summary"])
    
    # Enhanced tabs for all sections
    tabs = st.tabs([
        "🧠 Skills", 
        "💼 Experience", 
        "📁 Projects", 
        "🎓 Education",
        "📜 Certifications",
        "🏆 Achievements",
        "🌐 Languages",
        "🎯 Interests"
    ])
    
    # Skills Tab
    with tabs[0]:
        if parsed_info.get("skills"):
            # Create skill categories
            skill_categories = categorize_skills(parsed_info["skills"])
            
            for category, skills in skill_categories.items():
                if skills:
                    st.markdown(f"**{category}**")
                    skills_html = ""
                    for skill in skills:
                        skills_html += f'<span class="skill-tag">{skill}</span>'
                    st.markdown(skills_html, unsafe_allow_html=True)
                    st.write("")
        else:
            st.warning("No skills detected in your resume. Consider adding technical and soft skills.")
    
    # Experience Tab
    with tabs[1]:
        if parsed_info.get("experience"):
            st.text(parsed_info["experience"])
        else:
            st.warning("No experience information found. Add your work history to strengthen your resume.")

    # Projects Tab
    with tabs[2]:
        if parsed_info.get("projects"):
            st.text(parsed_info["projects"])
        else:
            st.warning("No projects found. Consider adding academic or personal projects.")

    # Education Tab
    with tabs[3]:
        if parsed_info.get("education"):
            st.text(parsed_info["education"])
        else:
            st.warning("No education information found. Add your educational background.")

    # Certifications Tab
    with tabs[4]:
        if parsed_info.get("certifications"):
            st.text(parsed_info["certifications"])
        else:
            st.info("No certifications found. Professional certifications can boost your profile!")

    # Achievements Tab
    with tabs[5]:
        if parsed_info.get("achievements"):
            st.text(parsed_info["achievements"])
        else:
            st.info("No achievements listed. Add awards, honors, or recognitions.")

    # Languages Tab
    with tabs[6]:
        if parsed_info.get("languages"):
            st.text(parsed_info["languages"])
        else:
            st.info("No languages specified. Multilingual abilities can be an asset!")

    # Interests Tab
    with tabs[7]:
        if parsed_info.get("interests"):
            st.text(parsed_info["interests"])
        else:
            st.info("No interests listed. Personal interests can show cultural fit!")

def categorize_skills(skills):
    """Categorize skills into different groups."""
    from resume_parser import SKILLS_DB
    
    categorized = {
        "Programming Languages": [],
        "Web Technologies": [],
        "Databases": [],
        "Tools & Platforms": [],
        "AI/ML": [],
        "Other": []
    }
    
    for skill in skills:
        skill_lower = skill.lower()
        categorized_flag = False
        
        for category, skill_list in SKILLS_DB.items():
            if skill_lower in [s.lower() for s in skill_list]:
                if category == "programming":
                    categorized["Programming Languages"].append(skill)
                elif category == "web":
                    categorized["Web Technologies"].append(skill)
                elif category == "database":
                    categorized["Databases"].append(skill)
                elif category == "tools":
                    categorized["Tools & Platforms"].append(skill)
                elif category == "ml_ai":
                    categorized["AI/ML"].append(skill)
                else:
                    categorized["Other"].append(skill)
                categorized_flag = True
                break
        
        if not categorized_flag:
            categorized["Other"].append(skill)
    
    # Remove empty categories
    return {k: v for k, v in categorized.items() if v}

def show_job_matching(parsed_info, raw_text, pinecone_handler):
    """Enhanced job matching with skill gap analysis."""
    st.header("🔍 Job Role Recommendations")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    with col1:
        match_threshold = st.slider("Minimum Match %", 0, 100, 60)
    with col2:
        top_k = st.selectbox("Number of Results", [3, 5, 10], index=1)
    with col3:
        sort_by = st.selectbox("Sort By", ["Match Score", "Skill Gap", "Recent"])
    
    try:
        with st.spinner(f"Finding top {top_k} matching jobs..."):
            similar_jobs = pinecone_handler.search_similar(raw_text, top_k=top_k)
        
        if similar_jobs:
            # Filter by threshold
            filtered_jobs = [job for job in similar_jobs if job['score'] >= match_threshold]
            
            if filtered_jobs:
                st.success(f"Found {len(filtered_jobs)} jobs matching your criteria!")
                
                # Create a summary chart
                job_scores = [job['score'] for job in filtered_jobs]
                job_titles = [job.get('metadata', {}).get('job_title', job['id']) for job in filtered_jobs]
                
                fig = px.bar(
                    x=job_scores, 
                    y=job_titles, 
                    orientation='h',
                    title="Job Match Scores",
                    labels={'x': 'Match Score (%)', 'y': 'Job Title'},
                    color=job_scores,
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed job cards
                for i, job in enumerate(filtered_jobs, 1):
                    meta = job.get("metadata", {})
                    
                    # Skill gap analysis
                    user_skills = parsed_info.get("skills", [])
                    job_requirements = meta.get("requirements", "")
                    skill_gap = analyze_skill_gap(user_skills, job_requirements)
                    
                    with st.expander(
                        f"#{i} {meta.get('job_title', job['id'])} at {meta.get('company', 'Unknown')} - {job['score']}% match",
                        expanded=(i <= 3)
                    ):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Company:** {meta.get('company', 'Unknown')}")
                            st.markdown(f"**Description:** {meta.get('description', '')}")
                            
                            # Requirements with highlighting
                            if meta.get('requirements'):
                                st.markdown("**Requirements:**")
                                st.write(meta.get('requirements', ''))
                        
                        with col2:
                            # Match metrics
                            st.metric("Overall Match", f"{job['score']}%")
                            st.metric("Skill Match", f"{skill_gap['match_percentage']}%")
                            st.metric("Skills Gap", len(skill_gap['missing_skills']))
                            
                            # Match level indicator
                            if job['score'] >= 80:
                                st.success("🟢 Excellent Match!")
                            elif job['score'] >= 65:
                                st.info("🟡 Good Match")
                            else:
                                st.warning("🟠 Fair Match")
                        
                        # Skill gap details
                        if skill_gap['missing_skills']:
                            st.markdown("**🎯 Skills to Develop:**")
                            missing_html = ""
                            for skill in skill_gap['missing_skills'][:5]:  # Show top 5
                                missing_html += f'<span style="background: #ffebee; color: #c62828; padding: 4px 8px; margin: 2px; border-radius: 4px; font-size: 12px;">{skill}</span>'
                            st.markdown(missing_html, unsafe_allow_html=True)
                        
                        # Apply button
                        col1, col2, col3 = st.columns([1, 1, 2])
                        with col1:
                            if st.button(f"💼 Apply", key=f"apply_{i}"):
                                st.success("Application link would open here!")
                        with col2:
                            if st.button(f"💾 Save", key=f"save_{i}"):
                                st.success("Job saved to your profile!")
            else:
                st.warning(f"No jobs found with match score above {match_threshold}%. Try lowering the threshold.")
        else:
            st.warning("No matching jobs found. Try updating your resume with more relevant skills.")
            
    except Exception as e:
        st.error(f"Error finding matching jobs: {str(e)}")
        st.info("Make sure Pinecone is properly configured and job data is uploaded.")

def analyze_skill_gap(user_skills: list, job_requirements: str) -> dict:
    """Analyze skill gaps between user profile and job requirements."""
    from resume_parser import ALL_SKILLS
    
    # Extract required skills from job requirements
    required_skills = set()
    job_req_lower = job_requirements.lower()
    
    for skill in ALL_SKILLS:
        if skill in job_req_lower:
            required_skills.add(skill)
    
    user_skills_lower = {skill.lower() for skill in user_skills}
    
    # Calculate matches and gaps
    matching_skills = user_skills_lower.intersection(required_skills)
    missing_skills = required_skills - user_skills_lower
    
    match_percentage = (len(matching_skills) / len(required_skills) * 100) if required_skills else 0
    
    return {
        "matching_skills": list(matching_skills),
        "missing_skills": list(missing_skills),
        "match_percentage": round(match_percentage, 2),
        "total_required": len(required_skills)
    }

def show_ai_coach(parsed_info, gemini_api):
    """Enhanced AI coaching with more interactive features."""
    st.header("🤖 AI Career Coach")
    
    # Coach personality selector
    coach_style = st.selectbox(
        "Choose your coach style:",
        ["Professional", "Friendly", "Motivational", "Direct"],
        help="This affects the tone of AI responses"
    )
    
    # Create tabs for different coaching aspects
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Resume Review", 
        "🎯 Career Path", 
        "💡 Improvement Tips",
        "🎨 Resume Rewrite",
        "❓ Ask Coach"
    ])
    
    with tab1:
        st.markdown("### 📋 Comprehensive Resume Review")
        if st.button("🔍 Analyze My Resume", type="primary"):
            with st.spinner("AI is conducting detailed analysis..."):
                analysis = gemini_api.analyze_resume(parsed_info, style=coach_style)
            
            if analysis["success"]:
                # Display analysis in structured format
                st.markdown("### 📊 Analysis Results")
                
                # Parse the response to extract sections
                response_text = analysis["text"]
                sections = parse_ai_response(response_text)
                
                # Display each section in an expander
                for section, content in sections.items():
                    with st.expander(f"{section}", expanded=True):
                        st.write(content)
            else:
                st.error(f"Analysis failed: {analysis['error']}")
    
        with tab2:
            st.markdown("### 🎯 Career Path Recommendations")
            
            col1, col2 = st.columns(2)
            with col1:
                experience_level = st.selectbox(
                    "Your Experience Level",
                    ["Entry Level", "Mid Level", "Senior Level", "Executive"]
                )
            with col2:
                career_goal = st.text_input("Your Career Goal (optional)", 
                                        placeholder="e.g., Data Scientist, ML Engineer")
            
            if st.button("🚀 Get Career Recommendations", type="primary"):
                with st.spinner("AI is mapping your career path..."):
                    enhanced_info = {
                        **parsed_info,
                        "experience_level": experience_level,
                        "career_goal": career_goal
                    }
                    suggestions = gemini_api.suggest_job_roles(enhanced_info)
                
                if suggestions["success"]:
                    st.markdown("### 🗺️ Your Career Roadmap")
                    st.write(suggestions["text"])
                    
                    # Create a career progression visualization
                    if "recommended job roles" in suggestions["text"].lower():
                        st.markdown("### 📈 Career Progression Timeline")
                        create_career_timeline(experience_level)
                else:
                    st.error(f"Recommendations failed: {suggestions['error']}")
        
    with tab3:
        st.markdown("### 💡 Personalized Improvement Tips")
        
        tip_category = st.selectbox(
            "Select Focus Area",
            ["Overall Resume", "Skills Section", "Experience Section", "ATS Optimization", "Industry-Specific"]
        )
        
        if st.button("💡 Get Improvement Tips", type="primary"):
            with st.spinner("AI is generating personalized tips..."):
                tips = gemini_api.generate_resume_tips(parsed_info, focus_area=tip_category)
            
            if tips["success"]:
                st.markdown("### 📝 Actionable Improvements")
                
                # Display tips in a nice format
                tips_text = tips["text"]
                tips_list = tips_text.split('\n')
                
                for tip in tips_list:
                    if tip.strip() and any(char.isdigit() for char in tip[:3]):
                        st.markdown(f"✅ {tip.strip()}")
            else:
                st.error(f"Tips generation failed: {tips['error']}")
    
    with tab4:
        st.markdown("### 🎨 AI-Powered Resume Rewrite")
        st.info("Let AI help you rewrite sections of your resume for maximum impact!")
        
        section_to_rewrite = st.selectbox(
            "Select Section to Enhance",
            ["Summary", "Experience Descriptions", "Skills", "Full Resume"]
        )
        
        writing_style = st.select_slider(
            "Writing Style",
            ["Concise", "Balanced", "Detailed"],
            value="Balanced"
        )
        
        if st.button("✨ Rewrite Section", type="primary"):
            with st.spinner(f"AI is rewriting your {section_to_rewrite.lower()}..."):
                rewritten = gemini_api.rewrite_section(
                    parsed_info, 
                    section_to_rewrite.lower(), 
                    writing_style
                )
            
            if rewritten["success"]:
                st.markdown("### 📝 Enhanced Version")
                st.success("Here's your professionally rewritten content:")
                
                # Show before/after comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original:**")
                    original_content = get_section_content(parsed_info, section_to_rewrite.lower())
                    st.text_area("", original_content, height=200, key="original")
                
                with col2:
                    st.markdown("**Enhanced:**")
                    st.text_area("", rewritten["text"], height=200, key="enhanced")
                
                if st.button("📋 Copy Enhanced Version"):
                    st.success("Copied to clipboard! (Feature requires additional JS)")
            else:
                st.error(f"Rewrite failed: {rewritten['error']}")
    
    with tab5:
        st.markdown("### ❓ Ask Your AI Career Coach")
        st.write("Have specific questions? Ask your AI coach anything about your career!")
        
        # Conversation history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat interface
        user_question = st.text_input(
            "Your question:",
            placeholder="e.g., How can I transition to data science?",
            key="user_question"
        )
        
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔍 Ask", type="primary"):
                if user_question:
                    with st.spinner("Coach is thinking..."):
                        response = gemini_api.answer_career_question(
                            user_question, 
                            parsed_info,
                            st.session_state.chat_history
                        )
                    
                    if response["success"]:
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "question": user_question,
                            "answer": response["text"]
                        })
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 💬 Conversation History")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                with st.expander(f"Q: {chat['question'][:50]}...", expanded=(i==0)):
                    st.markdown(f"**You asked:** {chat['question']}")
                    st.markdown(f"**Coach:** {chat['answer']}")

def show_dashboard(parsed_info, raw_text):
    """Enhanced analytics dashboard with more insights."""
    st.header("📊 Resume Analytics Dashboard")
    
    # Calculate various scores
    from resume_scorer import ResumeScorer
    scorer = ResumeScorer()
    overall_score, section_scores = scorer.calculate_score(parsed_info)
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = overall_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Score"},
            delta = {'reference': 70},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        ats_score = calculate_ats_compatibility(parsed_info, raw_text)
        st.metric("🤖 ATS Score", f"{ats_score}%", 
                  delta=f"+{ats_score-65}" if ats_score >= 65 else f"{ats_score-65}")
        st.progress(ats_score/100)
    
    with col3:
        word_count = len(raw_text.split())
        optimal = "✅" if 300 <= word_count <= 800 else "⚠️"
        st.metric("📝 Word Count", word_count, delta=f"{optimal} Optimal: 300-800")
    
    with col4:
        keyword_density = calculate_keyword_density(parsed_info, raw_text)
        st.metric("🔑 Keyword Density", f"{keyword_density:.1f}%", 
                  delta="Good" if 2 <= keyword_density <= 5 else "Adjust")
    
    # Section Analysis
    st.markdown("### 📊 Section Analysis")
    
    # Create radar chart for section scores
    categories = list(section_scores.keys())
    values = list(section_scores.values())
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Resume'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Resume Section Strength"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed section breakdown
    st.markdown("### 📋 Detailed Breakdown")
    
    for section, score in section_scores.items():
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"**{section.title()}**")
        with col2:
            st.progress(score/100)
        with col3:
            if score >= 80:
                st.success(f"{score}%")
            elif score >= 60:
                st.warning(f"{score}%")
            else:
                st.error(f"{score}%")
    
    # Improvement Recommendations
    st.markdown("### 🎯 Improvement Recommendations")
    recommendations = scorer.get_improvement_suggestions(section_scores)
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    else:
        st.success("Your resume is well-optimized! Minor tweaks can still help.")
    
    # Industry Comparison
    st.markdown("### 📊 Industry Comparison")
    
    # Mock industry average data
    industry_avg = {
        "Skills": 75,
        "Experience": 80,
        "Projects": 60,
        "Education": 85,
        "Overall": 75
    }
    
    comparison_data = pd.DataFrame({
        'Metric': list(industry_avg.keys()),
        'Your Score': [
            section_scores.get('skills', 0),
            section_scores.get('experience', 0),
            section_scores.get('projects', 0),
            section_scores.get('education', 0),
            overall_score
        ],
        'Industry Average': list(industry_avg.values())
    })
    
    fig = px.bar(
        comparison_data,
        x='Metric',
        y=['Your Score', 'Industry Average'],
        title="Your Resume vs Industry Average",
        barmode='group',
        color_discrete_map={'Your Score': '#667eea', 'Industry Average': '#764ba2'}
    )
    st.plotly_chart(fig, use_container_width=True)

def show_export_options(parsed_info, gemini_api):
    """Export resume in various formats."""
    st.header("💾 Export Your Resume")
    
    st.markdown("### Choose Export Format")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h4>📄 Enhanced PDF</h4>
            <p>Professional PDF with improved formatting</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Generate PDF", type="primary", key="pdf"):
            with st.spinner("Creating enhanced PDF..."):
                # This would integrate with a PDF generation library
                st.success("PDF generated! (Feature implementation needed)")
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h4>📝 ATS-Optimized</h4>
            <p>Plain text format optimized for ATS systems</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Generate ATS Version", type="primary", key="ats"):
            ats_text = generate_ats_resume(parsed_info)
            st.download_button(
                label="Download ATS Resume",
                data=ats_text,
                file_name="resume_ats_optimized.txt",
                mime="text/plain"
            )
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h4>💼 LinkedIn Ready</h4>
            <p>Formatted for LinkedIn profile sections</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Generate LinkedIn", type="primary", key="linkedin"):
            linkedin_text = generate_linkedin_sections(parsed_info)
            st.download_button(
                label="Download LinkedIn Content",
                data=linkedin_text,
                file_name="linkedin_profile.txt",
                mime="text/plain"
            )

# Helper functions
def parse_ai_response(response_text):
    """Parse AI response into structured sections."""
    sections = {}
    current_section = None
    current_content = []
    
    for line in response_text.split('\n'):
        if line.strip().startswith('**') and line.strip().endswith('**'):
            if current_section:
                sections[current_section] = '\n'.join(current_content)
            current_section = line.strip().replace('**', '')
            current_content = []
        else:
            if current_section:
                current_content.append(line)
    
    # Don't forget the last section
    if current_section:
        sections[current_section] = '\n'.join(current_content)
    
    return sections

def create_career_timeline(experience_level):
    """Create a visual career progression timeline."""
    timelines = {
        "Entry Level": ["Junior Developer", "Software Developer", "Senior Developer", "Tech Lead"],
        "Mid Level": ["Senior Developer", "Tech Lead", "Engineering Manager", "Director"],
        "Senior Level": ["Engineering Manager", "Director", "VP Engineering", "CTO"],
        "Executive": ["VP Engineering", "CTO", "Chief Innovation Officer", "Board Advisor"]
    }
    
    timeline = timelines.get(experience_level, timelines["Entry Level"])
    
    # Create timeline visualization
    fig = go.Figure()
    
    for i, role in enumerate(timeline):
        fig.add_trace(go.Scatter(
            x=[i*2, i*2+1.5],
            y=[1, 1],
            mode='lines+markers+text',
            name=role,
            text=['', role],
            textposition="top center",
            line=dict(color='rgb(102, 126, 234)', width=4),
            marker=dict(size=12),
            showlegend=False
        ))
    
    fig.update_layout(
        title="Potential Career Progression",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 2]),
        height=200,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def get_section_content(parsed_info, section):
    """Extract content from a specific section, now handling raw text."""
    section_key = section.lower().replace(" ", "_")
    
    if section_key == "summary":
        return parsed_info.get("summary", "No summary provided")
    elif section_key == "experience_descriptions":
        return parsed_info.get("experience", "No experience description found")
    elif section_key == "skills":
        return ", ".join(parsed_info.get("skills", [])) or "No skills listed"
    elif section_key == "full_resume":
        return st.session_state.get("raw_text", "No raw text found")
    else:
        # Generic handler for other sections
        return parsed_info.get(section_key, f"Section '{section}' not found")

def calculate_ats_compatibility(parsed_info, raw_text):
    """Calculate ATS compatibility score."""
    score = 0
    
    # Check for standard sections
    standard_sections = ["contact", "summary", "experience", "education", "skills"]
    for section in standard_sections:
        if section in parsed_info and parsed_info[section]:
            score += 15
    
    # Check for keywords
    keywords = len(parsed_info.get("skills", []))
    if keywords >= 10:
        score += 15
    elif keywords >= 5:
        score += 10
    
    # Check formatting (simple text, no complex formatting)
    if raw_text:
        # Penalize for special characters that might confuse ATS
        special_chars = len(re.findall(r'[█▪▫◆●]', raw_text))
        if special_chars < 5:
            score += 10
    
    # Contact information completeness
    contact = parsed_info.get("contact", {})
    if contact.get("email"):
        score += 5
    if contact.get("phone"):
        score += 5
    
    return min(score, 100)

def calculate_keyword_density(parsed_info, raw_text):
    """Calculate keyword density for the resume."""
    if not raw_text:
        return 0
    
    skills = parsed_info.get("skills", [])
    total_words = len(raw_text.split())
    
    if total_words == 0:
        return 0
    
    keyword_count = 0
    text_lower = raw_text.lower()
    
    for skill in skills:
        keyword_count += text_lower.count(skill.lower())
    
    density = (keyword_count / total_words) * 100
    return round(density, 2)

def generate_ats_resume(parsed_info):
    """Generate ATS-friendly plain text resume from raw sections."""
    lines = []
    
    # Contact Information
    contact = parsed_info.get("contact", {})
    if contact.get("name"):
        lines.append(contact["name"].upper())
    for key in ['email', 'phone', 'linkedin', 'github']:
        if contact.get(key):
            lines.append(f"{key.title()}: {contact[key]}")
    lines.append("\n" + "="*20 + "\n")
    
    # Section order
    section_order = ['summary', 'skills', 'experience', 'projects', 'education', 'certifications', 'achievements', 'languages', 'interests']
    
    for section_name in section_order:
        content = parsed_info.get(section_name)
        if content:
            lines.append(section_name.upper())
            lines.append("-" * len(section_name))
            if section_name == 'skills':
                lines.append(", ".join(content) if isinstance(content, list) else str(content))
            else:
                lines.append(str(content))
            lines.append("\n" + "="*20 + "\n")
            
    return "\n".join(lines)

def generate_linkedin_sections(parsed_info):
    """Generate content optimized for LinkedIn profile from raw sections."""
    sections = []
    
    sections.append("=== LINKEDIN PROFILE CONTENT ===\n")
    
    # About Section
    sections.append("ABOUT SECTION (SUMMARY):")
    sections.append(parsed_info.get("summary", "[Add a compelling professional summary here]"))
    sections.append("\n" + "="*20 + "\n")
    
    # Experience Section
    sections.append("EXPERIENCE SECTION:")
    sections.append(parsed_info.get("experience", "[Copy and paste your experience details here, formatting each job separately]"))
    sections.append("\n" + "="*20 + "\n")
    
    # Skills Section
    sections.append("SKILLS TO ADD (comma-separated):")
    sections.append(", ".join(parsed_info.get("skills", [])))
    
    return "\n".join(sections)

# Resume Scorer class (create this in a separate file ideally)
class ResumeScorer:
    """Score resumes based on various criteria, adapted for raw text."""
    
    def __init__(self):
        self.weights = {
            "skills": 0.25,
            "experience": 0.30,
            "education": 0.20,
            "projects": 0.15,
            "summary": 0.10
        }
    
    def _score_section_by_length(self, content, min_len=50, max_len=500, scale=1.0):
        """Helper to score a section based on content length."""
        if not content or not isinstance(content, str):
            return 0
        length = len(content)
        if length < min_len:
            return (length / min_len) * 50 * scale
        score = 50 + ((min(length, max_len) - min_len) / (max_len - min_len)) * 50
        return min(100, score * scale)

    def calculate_score(self, parsed_info):
        """Calculate overall resume score and section scores from raw text."""
        section_scores = {}
        
        # Score sections based on presence and content length
        section_scores["skills"] = min(100, len(parsed_info.get("skills", [])) * 8)
        section_scores["experience"] = self._score_section_by_length(parsed_info.get("experience", ""), min_len=100, max_len=1500)
        section_scores["education"] = self._score_section_by_length(parsed_info.get("education", ""), min_len=50, max_len=400)
        section_scores["projects"] = self._score_section_by_length(parsed_info.get("projects", ""), min_len=50, max_len=1000)
        section_scores["summary"] = self._score_section_by_length(parsed_info.get("summary", ""), min_len=50, max_len=500)
        
        # Calculate weighted overall score
        overall_score = sum(
            section_scores.get(section, 0) * weight
            for section, weight in self.weights.items()
        )
        
        # Add other sections to scores for radar chart, but don't weight them
        for section in ["certifications", "achievements", "languages"]:
            if section in parsed_info:
                section_scores[section] = self._score_section_by_length(parsed_info.get(section, ""), min_len=20, max_len=200)

        return overall_score, section_scores
    
    def get_improvement_suggestions(self, section_scores):
        """Generate suggestions based on scores."""
        suggestions = []
        
        if section_scores.get("skills", 0) < 50:
            suggestions.append("Your skills section is sparse. Add more relevant technical and soft skills (aim for 10+).")
        
        if section_scores.get("experience", 0) < 50:
            suggestions.append("Your experience section seems brief. Elaborate on your roles with specific responsibilities and quantifiable achievements (e.g., 'Increased efficiency by 20%').")
        
        if section_scores.get("education", 0) < 50:
            suggestions.append("Ensure your education section clearly lists your degrees, institutions, and graduation dates.")
        
        if section_scores.get("projects", 0) < 40:
            suggestions.append("Consider adding a projects section to showcase practical application of your skills. Describe 2-3 key projects.")
        
        if section_scores.get("summary", 0) < 50:
            suggestions.append("Write a compelling professional summary (2-4 sentences) at the top of your resume to grab the recruiter's attention.")
        
        if not suggestions:
            suggestions.append("Your resume has a solid foundation! Consider fine-tuning the language to be more action-oriented and results-focused.")

        return suggestions

if __name__ == "__main__":
    main()