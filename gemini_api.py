# gemini_api.py - Enhanced Version

"""
Handles interaction with Google Gemini API for LLM-powered resume analysis, job suggestions, and feedback.
Enhanced with new features: section rewriting, career Q&A, style-based responses.
"""

import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import json

load_dotenv()

class GeminiAPI:
    """
    Enhanced wrapper for Google Gemini API with advanced resume analysis and career coaching features.
    """
    _model = None

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        if GeminiAPI._model is None:
            GeminiAPI._model = genai.GenerativeModel('gemini-2.0-flash')
        self.model = GeminiAPI._model
        
        # Define coaching styles
        self.coaching_styles = {
            "Professional": "Respond in a formal, professional tone with industry-standard terminology.",
            "Friendly": "Be warm, encouraging, and conversational while maintaining professionalism.",
            "Motivational": "Be inspiring, positive, and focus on the candidate's potential and strengths.",
            "Direct": "Be concise, straightforward, and focus on actionable feedback without sugar-coating."
        }

    def generate_content(self, prompt: str, style: str = "Professional") -> Dict[str, Any]:
        """
        Generate content from Gemini API with style consideration.
        """
        try:
            # Add style instruction to prompt
            style_instruction = self.coaching_styles.get(style, self.coaching_styles["Professional"])
            full_prompt = f"{style_instruction}\n\n{prompt}"
            
            response = self.model.generate_content(full_prompt)
            return {"success": True, "text": response.text, "prompt_used": full_prompt}
        except Exception as e:
            return {"success": False, "error": str(e), "text": "Error occurred while generating content"}

    def analyze_resume(self, resume_data: Dict[str, Any], style: str = "Professional") -> Dict[str, Any]:
        """
        Enhanced resume analysis with comprehensive feedback.
        """
        # Extract all sections
        skills = ", ".join(resume_data.get("skills", [])) if resume_data.get("skills") else "None listed"
        experience = resume_data.get("experience", [])
        projects = resume_data.get("projects", [])
        education = resume_data.get("education", [])
        certifications = resume_data.get("certifications", [])
        summary = resume_data.get("summary", "No summary provided")
        contact = resume_data.get("contact", {})
        
        # Format experience
        exp_summary = []
        for exp in experience:
            if isinstance(exp, dict):
                exp_summary.append(
                    f"- {exp.get('job_title', 'N/A')} at {exp.get('company', 'N/A')} "
                    f"({exp.get('dates', 'N/A')}): {exp.get('description', 'No description')[:100]}..."
                )
            elif isinstance(exp, str):
                exp_summary.append(f"- {exp}")

        # Handle education, projects, certifications as lists of dicts
        def join_names(items, max_items=None):
            if not items:
                return 'None listed'
            if max_items:
                items = items[:max_items]
            return '; '.join([i.get('name', str(i)) if isinstance(i, dict) else str(i) for i in items])

        # Prepare unicode status for contact info
        email_status = "✓" if contact.get('email') else "✗"
        phone_status = "✓" if contact.get('phone') else "✗"
        linkedin_status = "✓" if contact.get('linkedin') else "✗"

        prompt = f"""
        You are an expert career coach and resume analyzer. Analyze this resume comprehensively.

        RESUME DETAILS:
        
        Contact Information:
        - Name: {contact.get('name', 'Not provided')}
        - Email: {email_status}
        - Phone: {phone_status}
        - LinkedIn: {linkedin_status}
        
        Professional Summary:
        {summary}
        
        Skills ({len(resume_data.get('skills', []))} listed):
        {skills}
        
        Experience ({len(experience)} positions):
        {chr(10).join(exp_summary[:5])}  # Show first 5
        
        Education ({len(education)} entries):
        {join_names(education, 3)}
        
        Projects ({len(projects)} projects):
        {join_names(projects, 3)}
        
        Certifications ({len(certifications)} certifications):
        {join_names(certifications, 3)}

        Please provide a comprehensive analysis with:
        
        1. **OVERALL IMPRESSION** (First impression and general assessment)
        
        2. **STRENGTHS** (Top 5 key strengths with specific examples)
        
        3. **AREAS FOR IMPROVEMENT** (5 specific, actionable improvements)
        
        4. **MISSING ELEMENTS** (Critical sections or information that's missing)
        
        5. **ATS OPTIMIZATION** (Specific tips for passing ATS systems)
        
        6. **COMPETITIVE EDGE** (What makes this candidate stand out or what's needed to stand out)
        
        7. **OVERALL SCORE** (Score out of 100 with detailed breakdown)
        
        Format your response with clear headers and bullet points. Be specific and actionable.
        """
        
        return self.generate_content(prompt, style)

    def suggest_job_roles(self, resume_data: Dict[str, Any], matched_jobs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Enhanced job role suggestions with career progression paths.
        """
        skills = ", ".join(resume_data.get("skills", []))
        experience = resume_data.get("experience", [])
        education = resume_data.get("education", [])
        projects = resume_data.get("projects", [])
        experience_level = resume_data.get("experience_level", "Entry Level")
        career_goal = resume_data.get("career_goal", "")
        
        # Calculate years of experience
        total_experience = len(experience)
        
        # Helper for joining names
        def join_names(items, max_items=None):
            if not items:
                return 'Not specified'
            if max_items:
                items = items[:max_items]
            return ', '.join([i.get('name', str(i)) if isinstance(i, dict) else str(i) for i in items])

        prompt = f"""
        You are a career counselor specializing in tech career paths. Based on this profile, provide comprehensive career guidance.

        CANDIDATE PROFILE:
        - Current Experience Level: {experience_level}
        - Career Goal: {career_goal if career_goal else "Not specified"}
        - Years of Experience: Approximately {total_experience} positions
        - Technical Skills: {skills}
        - Education: {join_names(education, 2)}
        - Projects: {len(projects)} technical projects
        
        Please provide:
        
        1. **IMMEDIATE OPPORTUNITIES** (5 job roles they can apply for now)
           - Include job title, typical requirements, and why they're a match
        
        2. **SHORT-TERM GOALS** (Next 6-12 months)
           - Skills to develop
           - Certifications to pursue
           - Projects to build
        
        3. **MEDIUM-TERM PROGRESSION** (1-3 years)
           - Natural career progression paths
           - Target roles and responsibilities
        
        4. **LONG-TERM VISION** (3-5 years)
           - Senior positions to aim for
           - Leadership opportunities
        
        5. **SKILL DEVELOPMENT ROADMAP**
           - Priority skills to learn
           - Learning resources and platforms
        
        6. **SALARY EXPECTATIONS**
           - Current market range for immediate opportunities
           - Progression potential
        
        7. **INDUSTRY RECOMMENDATIONS**
           - Best industries/sectors for this profile
           - Emerging opportunities
        
        Be specific with job titles, companies types, and actionable steps.
        """
        
        return self.generate_content(prompt)

    def generate_resume_tips(self, resume_data: Dict[str, Any], focus_area: str = "Overall Resume") -> Dict[str, Any]:
        """
        Generate specific tips based on focus area.
        """
        focus_prompts = {
            "Overall Resume": "Provide 10 comprehensive tips to improve all aspects of this resume.",
            "Skills Section": "Focus on optimizing the skills section with keyword optimization and organization tips.",
            "Experience Section": "Provide specific tips to enhance work experience descriptions with impact and metrics.",
            "ATS Optimization": "Give specific tips to ensure this resume passes ATS systems and reaches human recruiters.",
            "Industry-Specific": "Provide industry-specific tips based on the candidate's target roles and current market trends."
        }
        
        base_stats = f"""
        Current Resume Statistics:
        - Skills: {len(resume_data.get('skills', []))} listed
        - Experience: {len(resume_data.get('experience', []))} positions
        - Projects: {len(resume_data.get('projects', []))} projects
        - Education: {len(resume_data.get('education', []))} entries
        - Certifications: {len(resume_data.get('certifications', []))} certifications
        - Has Summary: {'Yes' if resume_data.get('summary') else 'No'}
        """
        
        prompt = f"""
        As a professional resume writer and career coach, provide specific, actionable tips.
        
        {base_stats}
        
        Focus Area: {focus_area}
        Task: {focus_prompts.get(focus_area, focus_prompts['Overall Resume'])}
        
        Requirements for your response:
        1. Number each tip clearly (1-10)
        2. Make each tip specific and actionable
        3. Include examples where relevant
        4. Prioritize high-impact improvements
        5. Consider current industry standards and ATS requirements
        
        Format: Clear numbered list with brief explanations and examples.
        """
        
        return self.generate_content(prompt)

    def rewrite_section(self, resume_data: Dict[str, Any], section: str, writing_style: str = "Balanced") -> Dict[str, Any]:
        """
        Rewrite a specific resume section with enhanced content.
        """
        style_guidelines = {
            "Concise": "Use bullet points, action verbs, and minimal words while maintaining impact.",
            "Balanced": "Provide comprehensive information with a good mix of detail and brevity.",
            "Detailed": "Include thorough descriptions with context, metrics, and comprehensive information."
        }
        
        section_content = self._get_section_content(resume_data, section)
        
        if not section_content:
            return {"success": False, "error": f"No content found for section: {section}"}
        
        prompt = f"""
        You are a professional resume writer. Rewrite the following resume section to maximize impact.
        
        Section to Rewrite: {section}
        Writing Style: {writing_style} - {style_guidelines[writing_style]}
        
        Current Content:
        {section_content}
        
        Requirements:
        1. Use strong action verbs and power words
        2. Include quantifiable metrics and achievements where possible
        3. Optimize for ATS scanning
        4. Maintain professional tone
        5. Highlight impact and value delivered
        6. Use industry-standard terminology
        7. For experience: Start each bullet with an action verb
        8. For summary: Create a compelling narrative in 3-4 sentences
        9. For skills: Organize by categories if applicable
        
        Provide only the rewritten content without any additional explanation.
        """
        
        return self.generate_content(prompt)

    def answer_career_question(self, question: str, resume_data: Dict[str, Any], chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Answer specific career questions with context from resume and chat history.
        """
        # Build context from resume
        skills = ", ".join(resume_data.get("skills", [])[:10])  # Top 10 skills
        experience_years = len(resume_data.get("experience", []))
        education = resume_data.get("education", [])
        current_role = "Not specified"
        
        if resume_data.get("experience") and len(resume_data["experience"]) > 0:
            latest_exp = resume_data["experience"][0]
            if isinstance(latest_exp, dict):
                current_role = latest_exp.get("job_title", "Not specified")
        
        # Build chat context (last 3 exchanges)
        chat_context = ""
        if chat_history and len(chat_history) > 0:
            chat_context = "Previous conversation context:\n"
            for chat in chat_history[-3:]:
                chat_context += f"Q: {chat['question']}\n"
                chat_context += f"A: {chat['answer'][:200]}...\n\n"
        
        prompt = f"""
        You are a helpful AI career coach having a conversation with a job seeker. Answer their question based on their profile and conversation history.
        
        CANDIDATE PROFILE:
        - Current/Recent Role: {current_role}
        - Years of Experience: ~{experience_years} positions
        - Key Skills: {skills}
        - Education: {education[0] if education else 'Not specified'}
        
        {chat_context}
        
        Current Question: {question}
        
        Guidelines:
        1. Provide a helpful, personalized answer based on their profile
        2. Be conversational but professional
        3. Include specific, actionable advice
        4. Reference their background when relevant
        5. Keep the response focused and not too long (2-3 paragraphs)
        6. If asked about something not in their profile, acknowledge that and provide general guidance
        
        Answer the question directly without any preamble.
        """
        
        return self.generate_content(prompt)

    def compare_with_job(self, resume_data: Dict[str, Any], job_description: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced job comparison with detailed analysis.
        """
        skills = resume_data.get("skills", [])
        experience = resume_data.get("experience", [])
        education = resume_data.get("education", [])
        projects = resume_data.get("projects", [])
        
        job_title = job_description.get("job_title", job_description.get("id", "Unknown"))
        job_reqs = job_description.get("requirements", "")
        job_desc = job_description.get("description", "")
        company = job_description.get("company", "Unknown")
        
        # Format experience summary
        exp_summary = []
        for exp in experience[:3]:  # Top 3 experiences
            if isinstance(exp, dict):
                exp_summary.append(f"- {exp.get('job_title', 'N/A')} at {exp.get('company', 'N/A')}")
        
        prompt = f"""
        You are an expert recruiter and ATS system. Compare this candidate's profile with the job requirements.
        
        JOB DETAILS:
        Position: {job_title} at {company}
        Description: {job_desc}
        Requirements: {job_reqs}
        
        CANDIDATE PROFILE:
        Skills: {', '.join(skills)}
        Experience: {len(experience)} positions including:
        {chr(10).join(exp_summary)}
        Education: {', '.join(education[:2]) if education else 'Not specified'}
        Projects: {len(projects)} technical projects
        
        Provide a detailed analysis including:
        
        1. **MATCH SCORE** (0-100%)
           - Overall compatibility score
           - Breakdown by category (skills, experience, education)
        
        2. **MATCHED QUALIFICATIONS**
           - Skills that directly match requirements
           - Relevant experience
           - Educational alignment
        
        3. **GAPS ANALYSIS**
           - Missing required skills
           - Experience gaps
           - Certification needs
        
        4. **COMPETITIVE ADVANTAGE**
           - What makes this candidate stand out
           - Unique qualifications
        
        5. **RED FLAGS**
           - Any concerns or mismatches
           - Overqualification risks
        
        6. **RECOMMENDATIONS**
           - How to improve candidacy
           - Skills to highlight in application
           - Interview preparation tips
        
        7. **HIRING PROBABILITY**
           - Likelihood of getting interview: High/Medium/Low
           - Reasoning for assessment
        
        Be specific and data-driven in your analysis.
        """
        
        result = self.generate_content(prompt)
        result["job_title"] = job_title
        result["company"] = company
        return result

    def _get_section_content(self, resume_data: Dict[str, Any], section: str) -> str:
        """
        Extract content from a specific section for rewriting.
        """
        section = section.lower()
        
        def join_names(items):
            return '\n'.join([i.get('name', str(i)) if isinstance(i, dict) else str(i) for i in items])

        if section == "summary":
            summary = resume_data.get("summary", "")
            if isinstance(summary, list):
                return "\n".join([s.get("name", str(s)) if isinstance(s, dict) else str(s) for s in summary])
            return str(summary)
        elif section == "skills":
            return ", ".join(resume_data.get("skills", []))
        elif section == "experience":
            exp_list = resume_data.get("experience", [])
            lines = []
            for exp in exp_list:
                if isinstance(exp, dict):
                    lines.append(f"{exp.get('job_title', 'Position')} at {exp.get('company', 'Company')} ({exp.get('dates', 'Dates')}): {exp.get('description', 'No description provided')}")
                else:
                    lines.append(str(exp))
            return "\n".join(lines)
        elif section == "education":
            return join_names(resume_data.get("education", []))
        elif section == "projects":
            return join_names(resume_data.get("projects", []))
        elif section == "certifications":
            return join_names(resume_data.get("certifications", []))
        elif section == "achievements":
            return join_names(resume_data.get("achievements", []))
        elif section == "languages":
            return join_names(resume_data.get("languages", []))
        else:
            # Fallback: try to join as string
            content = resume_data.get(section, "")
            if isinstance(content, list):
                return join_names(content)
            return str(content)

    def generate_interview_prep(self, resume_data: Dict[str, Any], job_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate interview preparation based on resume and job.
        """
        prompt = f"""
        You are an interview coach. Based on this candidate's profile and the job they're applying for, 
        provide comprehensive interview preparation.
        
        JOB: {job_info.get('job_title', 'Unknown')} at {job_info.get('company', 'Unknown')}
        
        CANDIDATE SKILLS: {', '.join(resume_data.get('skills', [])[:10])}
        EXPERIENCE: {len(resume_data.get('experience', []))} positions
        
        Provide:
        1. **TOP 10 LIKELY QUESTIONS** with suggested answers
        2. **TECHNICAL QUESTIONS** based on their skills
        3. **BEHAVIORAL QUESTIONS** using STAR method
        4. **QUESTIONS TO ASK** the interviewer
        5. **PREPARATION TIPS** specific to this role
        
        Make answers specific to their background.
        """
        
        return self.generate_content(prompt)

    def generate_cover_letter(self, resume_data: Dict[str, Any], job_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a tailored cover letter.
        """
        contact = resume_data.get("contact", {})
        skills = resume_data.get("skills", [])
        experience = resume_data.get("experience", [])
        
        # Get most recent experience
        recent_exp = "Entry level"
        if experience and isinstance(experience[0], dict):
            recent_exp = f"{experience[0].get('job_title', 'Professional')} with experience at {experience[0].get('company', 'various companies')}"
        
        prompt = f"""
        Write a compelling cover letter for this position.
        
        POSITION: {job_info.get('job_title', 'Unknown')} at {job_info.get('company', 'Unknown')}
        JOB DESCRIPTION: {job_info.get('description', '')}
        
        CANDIDATE:
        Name: {contact.get('name', '[Your Name]')}
        Current Status: {recent_exp}
        Key Skills: {', '.join(skills[:7])}
        
        Write a professional cover letter that:
        1. Opens with a strong hook
        2. Highlights relevant experience and skills
        3. Shows knowledge of the company
        4. Demonstrates value proposition
        5. Closes with a call to action
        
        Format it as a proper business letter.
        """
        
        return self.generate_content(prompt)

    def analyze_career_gap(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze career progression and identify gaps.
        """
        experience = resume_data.get("experience", [])
        skills = resume_data.get("skills", [])
        education = resume_data.get("education", [])
        
        prompt = f"""
        Analyze this candidate's career progression and identify gaps or areas for development.
        
        PROFILE:
        Experience: {len(experience)} positions
        Skills: {len(skills)} technical skills
        Education: {', '.join(education[:2]) if education else 'Not specified'}
        
        Provide:
        1. **CAREER TRAJECTORY ANALYSIS**
           - Current progression path
           - Consistency in career moves
           
        2. **SKILL GAPS**
           - Missing skills for next level
           - Outdated skills to update
           
        3. **EXPERIENCE GAPS**
           - Types of experience needed
           - Industries to explore
           
        4. **EDUCATION/CERTIFICATION GAPS**
           - Recommended certifications
           - Additional education needs
           
        5. **ACTION PLAN**
           - Immediate steps (next 30 days)
           - Short-term goals (3-6 months)
           - Long-term goals (1 year)
        """
        
        return self.generate_content(prompt)