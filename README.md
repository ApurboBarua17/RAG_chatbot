# UA CS Degree Planner

A RAG-powered chatbot built for University of Arizona CS students. As a CS 110 TA for 2+ years, I answered the same advising questions every semester from 150+ students — prerequisites, graduation requirements, track electives, CS vs CE differences. This tool gives students instant, grounded answers pulled directly from the UA course catalog and degree requirement documents, with no hallucination.

## Live Demo

[Coming soon — deploy to Render and paste URL here]

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Groq API — Llama 3.3 70B |
| Vector Store | Chroma (persisted locally) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Backend | FastAPI |
| Frontend | React + Tailwind CSS + Vite |
| Deployment | Render (free tier) |

## Local Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/ApurboBarua17/RAG_chatbot.git
   cd RAG_chatbot
   ```

2. **Add UA catalog PDFs** to `backend/data/`
   - CS undergraduate degree requirements (catalog.arizona.edu)
   - CS course descriptions (cs.arizona.edu)
   - CS advising FAQ
   - CS 110 syllabus
   - 4-year degree plan

3. **Get a free Groq API key** at console.groq.com (2 min, no credit card)

4. **Set up the backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   cp ../.env.example .env        # fill in GROQ_API_KEY
   python ingest.py               # builds the vector store (run once)
   uvicorn main:app --reload
   ```

5. **Start the frontend**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

Open http://localhost:5173

## How It Works

When you upload UA catalog PDFs, `ingest.py` chunks them into 500-token segments and embeds each chunk using a local sentence-transformer model, storing them in a persistent Chroma vector database. On each query, the FastAPI backend retrieves the 4 most semantically relevant chunks, injects them as context into a strict system prompt, and calls the Groq API (Llama 3.3 70B) to generate a grounded answer — with source citations and a hard refusal to fabricate course information.

## Screenshots

[Add screenshot after first run]
