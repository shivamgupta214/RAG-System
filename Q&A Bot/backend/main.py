import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Init FastAPI
app = FastAPI()

# Enable CORS for local React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === LangChain Setup ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_INDEX_NAME = "salesforce"

# Models
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)

# Pinecone Vector Store
vector_store = PineconeVectorStore(
    embedding=embedding_model,
    index_name=PINECONE_INDEX_NAME,
    namespace="salesforce",
    text_key="page_content"
)

# Custom Prompt
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert financial analyst assistant specializing in interpreting Salesforce earnings call transcripts.

You will receive context extracted from earnings call documents. Your job is to:
- Answer questions factually based only on the provided context.
- Provide clear, concise, and accurate answers.
- Do not guess or make up information — if the context doesn’t contain the answer, respond: "I couldn’t find that information in the documents."
- Where possible, mention the quarter or year if it adds clarity (e.g., Q4 2023).
- Keep responses concise, clear, and business-professional.
- If asked for summaries, focus on core business strategies, risk language, or leadership commentary.
- If asked for counts (e.g., # of documents or pages), infer from metadata in the context.

Context:
{context}

Question:
{question}

Answer:
"""
)

# Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)


# === WebSocket Route ===
@app.websocket("/ws/ask")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            await websocket.send_text("loading...")

            # Run LangChain RAG logic
            response = await asyncio.to_thread(qa_chain.invoke, {"query": data})
            print("\n\n 164",response)
            await websocket.send_text(response["result"])
        except Exception as e:
            await websocket.send_text(f"Error: {str(e)}")
            break
