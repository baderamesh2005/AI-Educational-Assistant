# AI Educational Assistant

An AI-powered educational assistant that uses **Retrieval-Augmented Generation (RAG)** to answer questions from uploaded educational PDFs and generate exam-style multiple-choice questions (MCQs).

## Features

- 📄 Upload an educational PDF
- 🔍 Extract and clean PDF content
- ✂️ Split documents into meaningful chunks
- 🧠 Generate embeddings using Sentence Transformers
- ⚡ Store and search embeddings using FAISS
- 🤖 Generate answers using Ollama or the optional Groq provider
- 📚 Ask questions based on the uploaded document
- 📝 Generate exactly 5 exam-style MCQs for a selected topic
- ✅ Interactive MCQ practice and instant grading
- 📖 Provide explanations for correct and incorrect answers
- 🔎 Retrieve relevant source information from the uploaded PDF
- 💻 React-based user interface
- 🐍 FastAPI backend

## System Architecture

```text
Student
   │
   ▼
Upload PDF
   │
   ▼
PDF Text Extraction
   │
   ▼
Text Cleaning
   │
   ▼
Text Chunking
   │
   ▼
Sentence Transformer Embeddings
   │
   ▼
FAISS Vector Store
   │
   ├───────────────┐
   ▼               ▼
 Ask AI         MCQ Practice
   │               │
   ▼               ▼
Retrieve        Retrieve
Relevant        Relevant
Chunks          Topic Content
   │               │
   ▼               ▼
Ollama LLM      Ollama LLM
   │               │
   ▼               ▼
Answer          5 MCQs
with Sources    + Explanations
Tech Stack
Frontend
React.js
Vite
Axios
HTML5
CSS3
Backend
Python
FastAPI
Uvicorn
AI / RAG
Ollama
Llama 3.2 3B
Sentence Transformers
FAISS
PyMuPDF
How It Works
1. PDF Upload

The user uploads a PDF through the React frontend.

The backend:

Extracts text using PyMuPDF
Cleans the extracted text
Splits the text into chunks
Generates vector embeddings
Stores the embeddings in FAISS
Makes the uploaded PDF the current knowledge source
2. Ask AI

When the student asks a question:

The question is converted into an embedding.
FAISS retrieves the most relevant document chunks.
Relevant context is provided to the local LLM.
Ollama generates an answer based on the retrieved content.
Source information is displayed with the answer.
3. MCQ Generation

When the student enters a topic:

Relevant content is retrieved from the current PDF.
The retrieved content is provided to the LLM.
The system generates 5 exam-style MCQs.
Each question contains 4 options.
Each question has exactly one correct answer.
An explanation is provided for the answer.
The student can attempt the quiz and receive their score.
Project Structure
AI-Educational-Assistant/
│
├── backend/
│   ├── chunker.py
│   ├── config.py
│   ├── document_processor.py
│   ├── embedding.py
│   ├── llm.py
│   ├── main.py
│   ├── mcq.py
│   ├── retriever.py
│   ├── text_cleaner.py
│   └── vector_store.py
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── tests/
├── requirements.txt
├── test_pipeline.py
├── test_suite_v2.py
├── README.md
└── .gitignore
Installation
Backend

Clone the repository:

git clone https://github.com/YOUR_USERNAME/AI-Educational-Assistant.git
cd AI-Educational-Assistant

Install Python dependencies:

pip install -r requirements.txt
Ollama

Install Ollama and make sure the required model is available:

ollama pull llama3.2:3b

Start the backend:

python -m uvicorn backend.main:app --reload --port 8008
Frontend

Open another terminal:

cd frontend
npm install
npm run dev

The frontend will normally be available at:

http://localhost:5173

The backend runs at:

http://127.0.0.1:8008
Requirements
Python 3.10+
Node.js
npm
Ollama
Llama 3.2 3B
Modern web browser

Provider and Deployment Configuration
The backend defaults to local Ollama and does not require a Groq API key. For Groq, set `LLM_PROVIDER=groq`, `GROQ_API_KEY`, and optionally `GROQ_MODEL_NAME`. For Ollama, set `LLM_PROVIDER=ollama` and optionally `OLLAMA_BASE_URL` or `OLLAMA_MODEL_NAME`.

For a deployed frontend, set `VITE_API_BASE_URL` to the backend's public API origin. Set backend `FRONTEND_ORIGINS` to the comma-separated frontend origins allowed by CORS. Local development keeps the frontend API default at `http://127.0.0.1:8008` and allows Vite origins on port 5173.

Answer Stream Events
`POST /ask/stream` sends Server-Sent Events whose `data` is a JSON object with a `type` field. `meta` carries source pages, confidence percentages, and retrieved sources; `token` carries an answer fragment; `error` carries a safe user-facing `message` and is terminal; `done` carries `status: "completed"` and timing data when generation completes. An `error` event is not followed by `done`.

Project Goal

The goal of this project is to provide students with an interactive AI learning assistant that can understand educational material from uploaded PDFs and help them:

Understand concepts
Ask questions
Get contextual answers
Practice exam-style MCQs
Learn from explanations
Prepare more effectively for examinations
Future Improvements
Multi-document knowledge bases
Conversation history
Better source highlighting
Voice-based interaction
More advanced question generation
Student performance analytics
Support for additional LLM models
License

This project is developed for educational and academic purposes.