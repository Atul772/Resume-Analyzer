# resume_scorer.py - Resume scoring and analysis utilities

from typing import Dict, List, Tuple
import re
from typing import Any


class ResumeScorer:
    """Score resumes based on various criteria."""
    
    def __init__(self):
        self.weights = {
            "skills": 0.20,
            "experience": 0.25,
            "education": 0.15,
            "projects": 0.15,
            "certifications": 0.10,
            "summary": 0.05,
            "formatting": 0.10
        }
        
        # Ideal counts for different sections
        self.ideal_counts = {
            "skills": (8, 15),  # (min, max)
            "experience": (2, 5),
            "projects": (2, 4),
            "education": (1, 3),
            "certifications": (1, 3)
        }
    
    def calculate_score(self, parsed_info: Dict) -> Tuple[float, Dict[str, float]]:
        """Calculate overall resume score and section scores."""
        section_scores = {}
        
        # Skills score
        skills_count = len(parsed_info.get("skills", []))
        min_skills, max_skills = self.ideal_counts["skills"]
        if skills_count < min_skills:
            section_scores["skills"] = (skills_count / min_skills) * 70
        elif skills_count <= max_skills:
            section_scores["skills"] = 70 + ((skills_count - min_skills) / (max_skills - min_skills)) * 30
        else:
            section_scores["skills"] = max(85, 100 - (skills_count - max_skills) * 2)  # Penalty for too many
        
        # Experience score
        experience = parsed_info.get("experience", [])
        exp_score = 0
        
        # Determine if it's a string (raw text) or list of dicts (structured data)
        if isinstance(experience, str):
            if not experience.strip():
                exp_score = 0
            else:
                # Estimate based on length and newlines indicating separate entries
                lines = [line for line in experience.split('\n') if line.strip()]
                # Base score for having experience text
                exp_score += 40
                
                # Check for dates (e.g. 2019-2021, Present, etc.) indicates structure
                import re
                if re.search(r'\b(201\d|202\d|199\d)\b', experience) or 'Present' in experience:
                    exp_score += 20
                
                # Length and details bonus
                words = len(experience.split())
                if words > 100:
                    exp_score += 40
                elif words > 50:
                    exp_score += 20
        else:
            # Handle list of dicts format
            for i, exp in enumerate(experience):
                if isinstance(exp, dict):
                    # Base points for having experience
                    exp_score += 20
                    
                    # Additional points for completeness
                    if exp.get("job_title") and exp["job_title"] != "Not specified": 
                        exp_score += 5
                    if exp.get("company") and exp["company"] != "Not specified": 
                        exp_score += 5
                    if exp.get("dates") and exp["dates"] != "Not specified": 
                        exp_score += 5
                    if exp.get("description") and exp["description"] != "No description provided":
                        # Check description quality
                        desc_length = len(exp["description"].split())
                        if desc_length > 30:
                            exp_score += 10
                        else:
                            exp_score += 5
                    
                    # Diminishing returns for too many experiences
                    if i >= 5:
                        exp_score = exp_score * 0.9

        section_scores["experience"] = min(100, exp_score)
        
        def _get_item_count(section_data):
            if isinstance(section_data, list):
                return len(section_data)
            elif isinstance(section_data, str):
                # Count non-empty lines starting with bullet points or dashes
                bullets = len([l for l in section_data.split('\n') if l.strip().startswith(('-', '•', '*'))])
                # Provide a reasonable estimation based on lines if no bullets
                lines = len([l for l in section_data.split('\n') if len(l.strip()) > 5])
                return max(bullets, max(1, lines // 3) if lines > 0 else 0)
            return 0

        # Education score
        education_data = parsed_info.get("education", [])
        education_count = _get_item_count(education_data)
        min_edu, max_edu = self.ideal_counts["education"]
        if not education_data:
            section_scores["education"] = 0
        elif education_count < min_edu:
            section_scores["education"] = 60
        else:
            section_scores["education"] = min(100, 60 + (education_count * 20))
        
        # Projects score
        projects_data = parsed_info.get("projects", [])
        projects_count = _get_item_count(projects_data)
        min_proj, max_proj = self.ideal_counts["projects"]
        if not projects_data:
            section_scores["projects"] = 0
        elif projects_count < min_proj:
            section_scores["projects"] = 60
        else:
            section_scores["projects"] = min(100, 60 + (projects_count * 15))
        
        # Certifications score
        cert_data = parsed_info.get("certifications", [])
        cert_count = _get_item_count(cert_data)
        if not cert_data:
            section_scores["certifications"] = 0
        else:
            section_scores["certifications"] = min(100, cert_count * 33.33)
        
        # Summary score
        summary = parsed_info.get("summary", "")
        if not summary:
            section_scores["summary"] = 0
        else:
            if isinstance(summary, list):
                summary_text = " ".join([s.get("name", str(s)) for s in summary if isinstance(s, dict)] + [str(s) for s in summary if not isinstance(s, dict)])
            else:
                summary_text = str(summary)
            summary_length = len(summary_text.split())
            if summary_length < 20:
                section_scores["summary"] = 30
            elif summary_length < 50:
                section_scores["summary"] = 60
            elif summary_length <= 150:
                section_scores["summary"] = 100
            else:
                section_scores["summary"] = max(70, 100 - (summary_length - 150) // 10)
        
        # Formatting score (based on successful parsing and section completeness)
        sections_found = len([k for k, v in parsed_info.items() if v and k != "sections_found"])
        total_possible_sections = 9  # contact, summary, skills, experience, education, projects, certifications, achievements, languages
        section_scores["formatting"] = (sections_found / total_possible_sections) * 100
        
        # Calculate weighted overall score
        overall_score = 0
        for section, weight in self.weights.items():
            overall_score += section_scores.get(section, 0) * weight
        
        return round(overall_score, 1), section_scores
    
    def get_improvement_suggestions(self, section_scores: Dict[str, float]) -> List[str]:
        """Generate prioritized suggestions based on scores."""
        suggestions = []
        
        # Priority 1: Critical missing sections (score = 0)
        if section_scores.get("skills", 0) == 0:
            suggestions.append("🚨 Add a skills section with 8-12 relevant technical and soft skills")
        
        if section_scores.get("experience", 0) == 0:
            suggestions.append("🚨 Add your work experience or internships with detailed descriptions")
        
        if section_scores.get("education", 0) == 0:
            suggestions.append("🚨 Include your educational background with degree and institution")
        
        # Priority 2: Low scoring sections (score < 50)
        if 0 < section_scores.get("skills", 0) < 50:
            suggestions.append("📈 Expand your skills section - aim for 8-12 relevant skills")
        
        if 0 < section_scores.get("experience", 0) < 50:
            suggestions.append("📈 Add more details to your work experience - include achievements and metrics")
        
        if section_scores.get("projects", 0) < 30:
            suggestions.append("📈 Add 2-3 relevant projects to showcase practical skills")
        
        if section_scores.get("summary", 0) < 60:
            suggestions.append("📈 Write a compelling professional summary (50-100 words)")
        
        # Priority 3: Good but can improve (score 50-80)
        if 50 <= section_scores.get("skills", 0) < 80:
            suggestions.append("💡 Organize skills by categories (Programming, Tools, Frameworks)")
        
        if section_scores.get("certifications", 0) < 30:
            suggestions.append("💡 Consider adding relevant certifications to boost credibility")
        
        # Priority 4: Formatting and structure
        if section_scores.get("formatting", 0) < 80:
            suggestions.append("🎨 Ensure all standard sections are present and properly labeled")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def calculate_ats_score(self, parsed_info: Dict, raw_text: str) -> int:
        """Calculate ATS (Applicant Tracking System) compatibility score."""
        ats_score = 0
        
        # 1. Standard sections presence (40 points)
        standard_sections = {
            "contact": 10,
            "summary": 5,
            "skills": 10,
            "experience": 10,
            "education": 5
        }
        
        for section, points in standard_sections.items():
            if section in parsed_info and parsed_info[section]:
                ats_score += points
        
        # 2. Contact information completeness (20 points)
        contact = parsed_info.get("contact", {})
        if contact.get("email"):
            ats_score += 8
        if contact.get("phone"):
            ats_score += 7
        if contact.get("name"):
            ats_score += 5
        
        # 3. Keywords and skills (20 points)
        skills_count = len(parsed_info.get("skills", []))
        if skills_count >= 5:
            ats_score += 10
        if skills_count >= 10:
            ats_score += 10
        
        # 4. Formatting simplicity (20 points)
        if raw_text:
            # Check for problematic elements
            special_chars = len(re.findall(r'[█▪▫◆●│└├]', raw_text))
            if special_chars < 5:
                ats_score += 10
            
            # Check for good structure (consistent spacing)
            lines = raw_text.split('\n')
            empty_lines = sum(1 for line in lines if not line.strip())
            if 10 <= empty_lines <= 30:  # Reasonable spacing
                ats_score += 10
        
        return min(100, ats_score)
    
    def get_section_feedback(self, section_name: str, section_score: float) -> Dict[str, Any]:
        """Get detailed feedback for a specific section."""
        feedback = {
            "score": section_score,
            "status": "Needs Improvement",
            "tips": []
        }
        
        # Determine status
        if section_score >= 90:
            feedback["status"] = "Excellent"
        elif section_score >= 70:
            feedback["status"] = "Good"
        elif section_score >= 50:
            feedback["status"] = "Fair"
        else:
            feedback["status"] = "Needs Improvement"
        
        # Section-specific tips
        if section_name == "skills":
            if section_score < 50:
                feedback["tips"] = [
                    "Add more technical skills relevant to your target role",
                    "Include both hard skills (programming, tools) and soft skills",
                    "Use industry-standard terminology"
                ]
            elif section_score < 80:
                feedback["tips"] = [
                    "Organize skills by categories",
                    "Prioritize most relevant skills first",
                    "Remove outdated or basic skills"
                ]
        
        elif section_name == "experience":
            if section_score < 50:
                feedback["tips"] = [
                    "Add quantifiable achievements to each role",
                    "Use strong action verbs to start bullet points",
                    "Include 3-5 bullet points per position",
                    "Focus on impact and results, not just duties"
                ]
            elif section_score < 80:
                feedback["tips"] = [
                    "Add more specific metrics and numbers",
                    "Highlight promotions or increased responsibilities",
                    "Tailor descriptions to your target role"
                ]
        
        elif section_name == "projects":
            if section_score < 30:
                feedback["tips"] = [
                    "Add 2-3 relevant projects that showcase your skills",
                    "Include project outcomes and technologies used",
                    "Link to GitHub or live demos if available"
                ]
        
        elif section_name == "education":
            if section_score < 50:
                feedback["tips"] = [
                    "Include degree, major, institution, and graduation date",
                    "Add relevant coursework if you're a recent graduate",
                    "Include GPA if it's 3.5 or higher"
                ]
        
        return feedback
    
    def calculate_competitiveness(self, parsed_info: Dict, target_level: str = "mid") -> Dict[str, Any]:
        """Calculate how competitive the resume is for different experience levels."""
        levels = {
            "entry": {
                "min_skills": 5,
                "min_experience": 0,
                "min_projects": 1,
                "education_weight": 0.4
            },
            "mid": {
                "min_skills": 10,
                "min_experience": 2,
                "min_projects": 2,
                "education_weight": 0.2
            },
            "senior": {
                "min_skills": 15,
                "min_experience": 3,
                "min_projects": 3,
                "education_weight": 0.1
            }
        }
        
        target_reqs = levels.get(target_level, levels["mid"])
        competitiveness_score = 0
        gaps = []
        
        # Check skills
        skills_count = len(parsed_info.get("skills", []))
        if skills_count >= target_reqs["min_skills"]:
            competitiveness_score += 30
        else:
            gap = target_reqs["min_skills"] - skills_count
            gaps.append(f"Add {gap} more skills")
            competitiveness_score += (skills_count / target_reqs["min_skills"]) * 30
        
        # Check experience
        exp_count = len(parsed_info.get("experience", []))
        if exp_count >= target_reqs["min_experience"]:
            competitiveness_score += 40
        else:
            if exp_count == 0:
                gaps.append("Add work experience or internships")
            else:
                gaps.append(f"Add {target_reqs['min_experience'] - exp_count} more experience entries")
            competitiveness_score += (exp_count / max(target_reqs["min_experience"], 1)) * 40
        
        # Check projects
        proj_count = len(parsed_info.get("projects", []))
        if proj_count >= target_reqs["min_projects"]:
            competitiveness_score += 20
        else:
            gaps.append(f"Add {target_reqs['min_projects'] - proj_count} more projects")
            competitiveness_score += (proj_count / target_reqs["min_projects"]) * 20
        
        # Education bonus
        if parsed_info.get("education"):
            competitiveness_score += 10
        
        return {
            "score": round(competitiveness_score, 1),
            "level": target_level,
            "gaps": gaps,
            "ready": competitiveness_score >= 70
        }