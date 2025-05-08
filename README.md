# 🤖 Salesforce Earnings Call Q&A App

This is a fullstack AI-powered Q&A application that allows users to ask questions about Salesforce earnings call transcripts using natural language. It uses:

- **FastAPI** for the backend
- **React.js** for the frontend
- **LangChain** for RAG (retrieval-augmented generation)
- **OpenAI GPT-4o** for answering questions
- **Pinecone** for vector search and semantic retrieval
- **Docker Compose** for easy container orchestration


## 📸 Demo Screenshot

<img width="1491" alt="Screenshot 2025-05-08 at 12 36 00 AM" src="https://github.com/user-attachments/assets/01b13283-d59c-4181-892d-a742535b9152" />

## 📸 System Design

![techstack](https://github.com/user-attachments/assets/2e44d07e-8e31-4c6f-8ba4-e7192165a852)

## 📸 Backend Flow

![backend-flow](https://github.com/user-attachments/assets/461bc440-f179-4b8f-9afa-630e12db993c)


---

## 🧠 Features

- Ask questions like:
  - “What are the risks Salesforce faced in 2023?”
  - “Summarize their Q1 2024 strategy.”
  - “How many pages are in the most recent earnings call?”
- Real-time chat UI using WebSocket
- Accurate, context-grounded answers using RAG
- Markdown rendering with bullet points and formatting

---

## 📦 Project Structure
project/
├── backend/ # FastAPI backend
│ ├── main.py # WebSocket + LangChain logic
│ ├── ingest.py # ⬅️ Loads PDFs, chunks, embeds to Pinecone
│ ├── requirements.txt # Python dependencies
│ ├── Dockerfile # Backend Docker image
│ └── .env # Environment variables
├── frontend/ # React frontend
│ ├── src/ # Components, UI logic
│ ├── public/
│ ├── Dockerfile # Frontend Docker image
│ └── package.json
├── docker-compose.yml # Runs both backend and frontend

---


## 🧾 `ingest.py` – Document Ingestion Pipeline

This script loads your **PDF transcripts**, **splits** them into chunks using LangChain’s `RecursiveCharacterTextSplitter`, and **embeds** them into **Pinecone** using `OpenAIEmbeddings`.

### ✅ How to use:

cd backend
python ingest.py

---

## 🔐 Environment Setup

### `.env` file inside `/backend`:

OPENAI_API_KEY=your-openai-api-key

PINECONE_API_KEY=your-pinecone-api-key

PINECONE_ENVIRONMENT=your-pinecone-environment

---
🚀 How to Run the App

git clone https://github.com/your-username/salesforce-qa-app.git

cd salesforce-qa-app

Run command: 
- docker-compose up --build  # ✅ This will start the FastAPI backend at http://localhost:8000 and start the React frontend at http://localhost:3000

  
Rebuild only frontend - docker-compose build frontend
Rebuild only backend - docker-compose build backend
Restart clean - docker-compose down && docker-compose up --build


