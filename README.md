# 🧠 Smart Career Coach: Resume Analyzer & Job Role Recommender

An AI-powered web app that analyzes resumes, provides personalized improvement tips, and recommends job roles using LLMs, NLP, and vector embeddings.

---

## 🚀 Live Demo

🔗 [Streamlit Cloud App (Coming Soon)](https://share.streamlit.io/...)

---

## 📌 Project Overview

**Smart Career Coach** helps job seekers improve their resumes and discover relevant job opportunities. It combines:

- 📄 Resume Parsing with NLP
- 🔍 Job Role Matching using semantic search
- 🤖 AI Suggestions using Gemini LLM
- 🧠 RAG (Retrieval-Augmented Generation) pipeline for job matching

---

## 🛠️ Features

| Feature | Description |
|--------|-------------|
| 📤 Resume Uploader | Upload your resume in PDF format |
| 📊 Resume Parser | Extracts skills, education, experience using `pdfplumber` + `spaCy` |
| 🔍 Job Role Recommender | Embeds job descriptions using `all-mpnet-base-v2` and queries via Pinecone |
| 🤖 LLM Feedback (Gemini 1.5) | Suggests improvements and suitable roles using generative AI |
| 💬 Chat Assistant (Optional) | Ask: "What jobs match my skills?" or "How can I improve my resume?" |
| 🌐 Deployed | Easily hosted on Streamlit Cloud (free) |

---

## 🧩 Technologies Used

| Category | Tools |
|---------|-------|
| Frontend | Streamlit |
| NLP & Embedding | spaCy, pdfplumber, sentence-transformers (`all-mpnet-base-v2`) |
| Vector Search | Pinecone |
| LLM Integration | Google Gemini API (1.5 Flash) |
| Backend / Hosting | Python, Streamlit Cloud |
| Secrets Management | `.env`, `python-dotenv` |

---

## 📂 Folder Structure

```
Resume Analyzer/
│
├── app.py                   # Streamlit frontend
├── resume_parser.py         # Extract info from PDF
├── pinecone_handler.py      # Pinecone setup and query functions
├── job_embedder.py          # Load and embed job descriptions
├── gemini_api.py            # Calls Gemini 1.5 API
├── requirements.txt         # Python dependencies
├── .env                     # API keys and secrets (not pushed)
├── .gitignore
├── README.md
├── sample_data/
│   ├── sample_resume.pdf
│   └── job_descriptions/
│       ├── data_scientist.txt
│       ├── ml_engineer.txt
│       └── analyst.txt
```

---

## 🧠 How It Works

1. ✅ User uploads resume → parsed into structured format
2. ✅ Resume is embedded → using Sentence Transformers
3. ✅ Job descriptions are embedded → stored in Pinecone
4. ✅ Semantic search → find top 3 closest job matches
5. ✅ Gemini API → suggests improvements + role matches
6. ✅ UI displays results in an interactive Streamlit app

---

## ⚙️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/your-username/resume-analyzer.git
cd resume-analyzer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Setup `.env` with API keys
```env
GEMINI_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_ENV=us-east-1-aws
PINECONE_INDEX=resume-analyser
```

### 4. Run the app locally
```bash
streamlit run app.py
```

---

## 🧪 Example Prompts (LLM)

**Resume Improvements Prompt:**
```
You're a career coach. Given this resume, suggest 3 improvements and 3 relevant job roles.
```

**JD Matching Prompt:**
```
Compare this resume with the following job description. Highlight matching skills and give a match score out of 100.
```

---

## 📸 Screenshots

_Add screenshots of Streamlit UI, parsed resume, matched roles, and Gemini suggestions here._

---

## 👨‍💼 Built For

- Final Year B.Tech Project
- Resume Analyzer + Job Matcher Demo for Placements
- Real-world use case combining NLP + LLMs + Vector Search

---

## 👨‍💻 Author

**Atul Kumar**  
[LinkedIn](https://linkedin.com/in/...)  
[GitHub](https://github.com/...)

---

## 📃 License

This project is licensed under the MIT License.