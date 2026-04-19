# ProjectHub AI Service

A standalone Python FastAPI microservice. Runs on port **8001**.
Your Node.js server calls it internally — no internet needed.

## Setup (Run once)

```bash
cd ai-service
pip install -r requirements.txt
```

## Start the AI Service

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

## Endpoints

| Method | Route | Feature |
|--------|-------|---------|
| POST | /plagiarism | TF-IDF + Cosine Similarity |
| POST | /summarize | Extractive Report Summarizer |
| POST | /chatbot | Intent-based NLP Chatbot |
| POST | /risk-predict | Weighted ML Risk Predictor |
| GET | /health | Health check |

## Running Both Servers (Development)

Open **two terminals**:

**Terminal 1 — AI Service:**
```bash
cd ai-service
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

**Terminal 2 — Node.js:**
```bash
cd server
npm run dev
```

Your React frontend talks to Node.js (port 4000),
Node.js talks to Python (port 8001) internally.

## Architecture

```
React (5173) → Node.js (4000) → Python AI (8001)
                     ↓
                  MongoDB
```
