# 🧠 Smart Career Coach: AI-Powered Resume Analyzer & Job Matcher

An advanced AI-powered web application that analyzes resumes, provides personalized career coaching, and matches candidates with relevant job opportunities using state-of-the-art NLP, LLMs, and vector search technologies.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 Features

- **Multi-format Resume Analysis:** Upload PDF, DOCX, or TXT resumes and extract all key sections (contact, skills, experience, education, projects, certifications, achievements, languages, interests).
- **ATS Compatibility & Resume Scoring:** Get an ATS score and detailed feedback for each section, with actionable improvement suggestions.
- **Intelligent Job Matching:** Semantic search using advanced embeddings, skill gap analysis, and hybrid search for best-fit job recommendations.
- **AI Career Coach (Google Gemini):** Choose your coaching style, get resume reviews, career path advice, section rewriting, Q&A, interview prep, and cover letter generation.
- **Analytics Dashboard:** Visualize your resume's strengths, keyword density, and competitiveness against industry standards.
- **Export Options:** Download ATS-optimized resumes, LinkedIn-ready content, and (soon) enhanced PDFs.

---

## 🛠️ Technology Stack

| Category         | Technologies                                      |
|------------------|--------------------------------------------------|
| **Frontend**     | Streamlit, Plotly                                |
| **NLP & Parsing**| spaCy, NLTK, pdfplumber, PyPDF2, python-docx     |
| **Embeddings**   | Sentence-Transformers (all-mpnet-base-v2)        |
| **Vector DB**    | Pinecone                                          |
| **LLM**          | Google Gemini 2.0 Flash                          |
| **Backend**      | Python 3.8+                                      |
| **Caching**      | LRU Cache, Disk-based embedding cache            |

---

## 📁 Project Structure

```
Resume Analyzer/
├── app.py                # Main Streamlit application
├── resume_parser.py      # Advanced resume parsing with NLP
├── pinecone_handler.py   # Vector database operations
├── job_embedder.py       # Job description processing
├── gemini_api.py         # LLM integration for AI coaching
├── embedding_utils.py    # Advanced embedding utilities
├── resume_scorer.py      # Resume scoring and analysis
├── requirements.txt      # Python dependencies
├── .env.example          # Example environment variables
├── README.md             # This file
├── job_description/      # Job descriptions directory
└── .embedding_cache/     # Cached embeddings (auto-created)
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- Pinecone account (free tier available)
- Google Gemini API key (free tier available)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/smart-career-coach.git
cd smart-career-coach
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Pinecone Configuration
PINECONE_ENV=us-east-1-aws
PINECONE_INDEX=resume-analyzer
```

### 5. Initialize Pinecone and Load Job Descriptions

**Option 1: Using the setup script**
```bash
python setup.py
```

**Option 2: Manual initialization**
```bash
python -c "from pinecone_handler import PineconeHandler; handler = PineconeHandler()"
python job_embedder.py
```

### 6. Run the Application

```bash
streamlit run app.py
```
The app will open in your browser at [http://localhost:8501](http://localhost:8501).

---

## 📝 Adding Custom Job Descriptions

1. Create a `.txt` file in the `job_description/` folder with this format:
    ```
    Job Title Here

    Company: Company Name

    Job Description:
    Detailed description of the role...

    Responsibilities:
    - First responsibility
    - Second responsibility

    Requirements:
    - First requirement
    - Second requirement
    ```
2. Run:
    ```bash
    python job_embedder.py
    ```

---

## 🎯 Usage Guide

1. **Upload Your Resume:**  
   Click "Upload your resume" and select a PDF, DOCX, or TXT file. The system will parse and analyze your resume.

2. **Explore Modes:**
   - **Resume Analysis:** View extracted info, scores, suggestions, and ATS compatibility.
   - **Job Matching:** Get top job matches, see skill gaps, filter by match %, and save/apply.
   - **AI Career Coach:** Choose coaching style, get reviews, career path, rewrite sections, and ask questions.
   - **Dashboard:** Visual analytics, industry comparison, improvement tracking, and export reports.
   - **Export:** Download ATS-optimized, LinkedIn-ready, or enhanced PDF resumes.

---

## 🔧 Configuration & Tips

- **Embedding Cache:**  
  The system caches embeddings for speed. To clear cache:
  ```python
  from embedding_utils import clear_cache
  clear_cache()
  ```

- **Search Parameters:**  
  Adjust in `app.py` (e.g., `match_threshold`, `top_k`).

- **Model Selection:**  
  Change the embedding model in `embedding_utils.py`.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the MIT License.