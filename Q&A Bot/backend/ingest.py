from langchain_community.document_loaders import PyMuPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "salesforce"

embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0.1, model_name="gpt-4.1", openai_api_key=OPENAI_API_KEY)

#Create a Pinecone index
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
if PINECONE_INDEX_NAME not in pc.list_indexes():
    print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(name=PINECONE_INDEX_NAME, dimension=1536, metric="cosine", spec=ServerlessSpec(
    cloud="aws",
    region="us-east-1",
  ),)
    print(f"Pinecone index '{PINECONE_INDEX_NAME}' created.")
else:
    print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")


pdfs = [os.path.join(root, file)
        for root, _, files in os.walk('./Earnings Call Transcripts')
        for file in files if file.endswith('.pdf')]

docs = []
for pdf in pdfs:
    loader = PyMuPDFLoader(pdf)
    pages = loader.load()

    docs.extend(pages)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""])

chunks = text_splitter.split_documents(docs)


#document vector embedding
batch_size = 20

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    vector_store = PineconeVectorStore.from_documents(documents=batch, embedding=embeddings_model, index_name=PINECONE_INDEX_NAME, namespace="salesforce", text_key="page_content")

