# resume_parser.py - Enhanced Version

import pdfplumber
import PyPDF2
import docx
import re
import nltk
from io import BytesIO

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Enhanced skills database with variations
SKILLS_DB = {
    "programming": ["python", "java", "c++", "c#", "c", "javascript", "typescript", "go", "rust", "scala", "kotlin", "ruby", "php", "swift", "dart", "bash", "shell", "r", "perl", "objective-c", "matlab"],
    "web": ["html", "css", "html5", "css3", "react", "react.js", "angular", "vue", "vue.js", "node.js", "express", "flask", "django", "fastapi", "spring", "spring boot", "next.js", "nuxt.js", "graphql", "jquery", "bootstrap", "tailwind", "tailwind css", "sass", "less"],
    "database": ["mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle", "sql", "nosql", "dynamodb", "elasticsearch", "cassandra", "mariadb", "firebase", "snowflake", "neo4j", "supabase", "pinecone", "pinecone-db"],
    "tools": ["git", "github", "gitlab", "bitbucket", "docker", "kubernetes", "jenkins", "aws", "azure", "gcp", "linux", "unix", "terraform", "ansible", "jira", "confluence", "webpack", "npm", "yarn", "postman", "figma", "vs code", "vscode"],
    "ml_ai": ["machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn", "scikit", "pandas", "numpy", "opencv", "nlp", "llm", "llms", "huggingface", "keras", "xgboost", "matplotlib", "seaborn", "nltk", "spacy", "generative ai", "data analysis", "langchain", "gradio", "rag"],
    "other": ["agile", "scrum", "rest", "api", "microservices", "devops", "ci/cd", "testing", "automation", "system design", "data structures", "algorithms", "problem solving", "ui/ux", "project management", "kanban", "communication", "teamwork", "leadership"]
}

# Flatten skills for easy lookup
ALL_SKILLS = set()
for category in SKILLS_DB.values():
    ALL_SKILLS.update([skill.lower() for skill in category])

class ResumeParser:
    def __init__(self):
        self.sections = {
            'contact': ['contact', 'personal information', 'personal details'],
            'summary': ['summary', 'objective', 'profile', 'about me', 'professional summary', 'career objective'],
            'experience': ['experience', 'work experience', 'employment', 'professional experience', 'work history', 'employment history'],
            'education': ['education', 'academic', 'qualification', 'academic background', 'academic qualifications'],
            'skills': ['skills', 'technical skills', 'competencies', 'core competencies', 'technical competencies', 'expertise'],
            'projects': ['projects', 'personal projects', 'academic projects', 'key projects'],
            'certifications': ['certifications', 'certificates', 'certification', 'professional certifications'],
            'achievements': ['achievements', 'accomplishments', 'awards', 'honors', 'recognition'],
            'languages': ['language', 'languages known', 'language proficiency', 'linguistic skills'],
            'interests': ['interests', 'hobbies', 'activities', 'extracurricular']
        }
    
    def extract_text_from_pdf(self, file):
        text = ""
        try:
            # Try with pdfplumber first
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e_plumber:
            print(f"pdfplumber failed: {e_plumber}, falling back to PyPDF2")
            # Fallback to PyPDF2
            try:
                # Reset file pointer if it's a file-like object
                if hasattr(file, 'seek'):
                    file.seek(0)
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            except Exception as e_pypdf:
                print(f"Error reading PDF with PyPDF2: {str(e_pypdf)}")
        return text
    
    def extract_text_from_docx(self, file):
        try:
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error reading DOCX: {str(e)}")
            return ""

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from any supported file format"""
        if file_path.lower().endswith('.pdf'):
            with open(file_path, 'rb') as f:
                return self.extract_text_from_pdf(f)
        elif file_path.lower().endswith('.docx'):
            with open(file_path, 'rb') as f:
                return self.extract_text_from_docx(f)
        elif file_path.lower().endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            except Exception as e:
                print(f"Error reading text file: {e}")
                return ""
        else:
            return ""
    
    def extract_contact_info(self, text):
        contact_info = {}
        
        # Email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact_info['email'] = email_match.group()
        
        # Phone
        phone_pattern = r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,5}[-\s\.]?[0-9]{1,5}'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            contact_info['phone'] = phone_match.group()
        
        # LinkedIn
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin_match = re.search(linkedin_pattern, text, re.IGNORECASE)
        if linkedin_match:
            contact_info['linkedin'] = linkedin_match.group()
        
        # GitHub
        github_pattern = r'github\.com/[\w-]+'
        github_match = re.search(github_pattern, text, re.IGNORECASE)
        if github_match:
            contact_info['github'] = github_match.group()
        
        # Name (usually first non-empty line)
        lines = text.split('\n')
        for line in lines[:5]:
            if line.strip() and len(line.strip()) < 50:
                if not any(char.isdigit() for char in line) and '@' not in line:
                    contact_info['name'] = line.strip()
                    break
        
        return contact_info
    
    def identify_sections(self, text):
        lines = text.split('\n')
        sections_content = {}
        current_section = None
        current_content = []
        
        # Create a set of all keywords for faster checking if a line could be a header
        all_keywords = set()
        for kw_list in self.sections.values():
            all_keywords.update(kw_list)

        for i, line in enumerate(lines):
            line_clean = line.strip()
            line_lower = line_clean.lower()
            
            # Skip empty lines but preserve them in the content
            if not line_clean:
                if current_section:
                    current_content.append(line)
                continue
            
            # Check if line is a section header
            is_section_header = False
            matched_section = None
            
            # Optimization: only check for section headers if the line is short and contains a keyword
            if len(line_clean) < 30 and any(kw in line_lower for kw in all_keywords):
                for section_type, keywords in self.sections.items():
                    for keyword in keywords:
                        # More robust checking for headers
                        if (line_lower == keyword or 
                            line_lower == keyword + ':' or
                            line_lower.startswith(keyword + ' ') or
                            (len(line_clean.split()) <= 3 and keyword in line_lower)):
                            
                            matched_section = section_type
                            is_section_header = True
                            break
                    if is_section_header:
                        break
            
            # Special handling for skills vs languages
            if matched_section == 'languages' and any(tech_word in line_lower for tech_word in 
                ['python', 'java', 'javascript', 'html', 'css', 'sql', 'react', 'node', 'git', 'framework']):
                matched_section = 'skills'
            
            if is_section_header and matched_section:
                # Save previous section
                if current_section and current_content:
                    sections_content[current_section] = '\n'.join(current_content).strip()
                
                current_section = matched_section
                current_content = [] # Don't include the header in the content
            else:
                # Add to current section content
                if current_section:
                    current_content.append(line)
        
        # Save the last section
        if current_section and current_content:
            sections_content[current_section] = '\n'.join(current_content).strip()
        
        return sections_content

    def extract_skills(self, section_text: str, is_skills_section: bool = False) -> list:
        """Extract skills from skills section text."""
        if not section_text:
            return []
        
        skills = set()
        text_lower = section_text.lower()
        
        # Use regex to find matches for whole words/phrases
        for skill in ALL_SKILLS:
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                skills.add(skill.title())
                
        # If this is specifically from a skills section, try to extract custom skills
        # by splitting on common delimiters (commas, bullets, pipes, newlines)
        if is_skills_section:
            lines = section_text.split('\n')
            for line in lines:
                # Remove leading bullets, dashes, asterisks
                cleaned_line = re.sub(r'^[\s•\-\*]+', '', line).strip()
                if not cleaned_line:
                    continue
                # Split by commas, pipes, bullets, or colons
                parts = re.split(r'[,|•:]', cleaned_line)
                for part in parts:
                    part = part.strip()
                    # Filter out category names often followed by colons if they accidentally sneak through
                    if part.lower() in ['programming languages', 'problem solving', 'tools & frameworks', 'web development', 'data science & ml', 'databases & os', 'technical skills']:
                        continue
                    # Filter out anything that's too long to be a single skill or too short
                    if 1 < len(part) <= 35:
                        part = re.sub(r'[.:;]$', '', part).strip()
                        skills.add(part.title())
        
        return sorted(list(skills))

    def parse_resume(self, file_path: str) -> dict:
        """
        Main method to parse a resume.
        It extracts text, identifies sections, and then returns the raw text of each section.
        It also extracts contact info and a list of skills.
        """
        text = self.extract_text_from_file(file_path)
        if not text:
            return {}

        contact_info = self.extract_contact_info(text)
        sections = self.identify_sections(text)

        # The new logic returns raw text for each section.
        # We will also parse the skills section specifically for use in other parts of the app.
        if 'skills' in sections:
            skills_list = self.extract_skills(sections['skills'], is_skills_section=True)
        else:
            # If no skills section, try to find skills in the whole text
            skills_list = self.extract_skills(text, is_skills_section=False)

        parsed_data = {
            'contact': contact_info,
            'skills': skills_list,
            'raw_text': text
        }
        
        # Add all identified sections to the parsed data
        parsed_data.update(sections)

        return parsed_data