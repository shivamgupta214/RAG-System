# import os
# import PyPDF2
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Pinecone
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv
# from pinecone import Pinecone
# from langchain_pinecone import PineconeVectorStore


# # CONFIGURATION
# load_dotenv()
# INDEX_NAME = "salesforce-earnings"
# PDF_FOLDER = "./Earnings Call Transcripts"

# # Setup API keys
# openai_key = os.environ.get("OPENAI_API_KEY")
# pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


# des = pc.describe_index(INDEX_NAME)
# print("\n\n line 20", des)

# index = pc.Index(host="https://salesforce-earnings-meectpm.svc.aped-4627-b74a.pinecone.io")

# # Initialize LangChain components
# embeddings = OpenAIEmbeddings()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# # Helper: extract text from PDF
# def extract_pdf_text(file_path):
#     text = ""
#     with open(file_path, 'rb') as f:
#         reader = PyPDF2.PdfReader(f)
#         for page in reader.pages:
#             text += page.extract_text()
#     return text

# # Ingest all PDFs
# for filename in os.listdir(PDF_FOLDER):
#     if filename.endswith(".pdf"):
#         pdf_path = os.path.join(PDF_FOLDER, filename)
#         print(f"Processing {filename}...")
        
#         full_text = extract_pdf_text(pdf_path)
#         chunks = text_splitter.split_text(full_text)
        
#         metadatas = [{"source": filename}] * len(chunks)
        
#         vectorstore = PineconeVectorStore.from_texts(
#             texts=chunks,
#             embedding=embeddings,
#             metadatas=metadatas,
#             index_name=INDEX_NAME
#         )
        
#         print(f"Uploaded {len(chunks)} chunks from {filename}.")

# print("✅ All PDFs ingested into Pinecone!")


# ===========================================

# import os
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Pinecone
# from pinecone import Pinecone, ServerlessSpec
# import glob  # Import the glob module
# from langchain_pinecone import PineconeVectorStore
# from langchain.document_loaders import NotionDirectoryLoader
# from langchain.text_splitter import CharacterTextSplitter
# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
# PINECONE_INDEX_NAME = "salesforce-ear"  # Replace with your Pinecone index name

# # Initialize OpenAI Embeddings
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# # Initialize Pinecone connection
# pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


# # if PINECONE_INDEX_NAME not in pc.list_indexes():
# #     print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
# #     pc.create_index(name=PINECONE_INDEX_NAME, dimension=1536, metric="cosine", spec=ServerlessSpec(
# #     cloud="aws",
# #     region="us-east-1",
# #   ),)
# #     print(f"Pinecone index '{PINECONE_INDEX_NAME}' created.")
# # else:
# #     print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

# des = pc.describe_index(PINECONE_INDEX_NAME)
# print("\n\n line 93", des)
# host = des['host']
# print("\n\n line 99", pc.list_indexes())
# index = pc.Index(host=host)

# def load_and_process_pdfs(pdf_folder):
#     """Loads and processes all PDF files in the specified folder."""
#     documents = []
#     pdf_paths = glob.glob(os.path.join(pdf_folder, "*.pdf"))  # Find all .pdf files

#     if not pdf_paths:
#         print(f"No PDF files found in the folder: {pdf_folder}")
#         return []

#     for path in pdf_paths:
#         print(f"Loading PDF: {path}")
#         loader = PyPDFLoader(path)
#         documents.extend(loader.load())

#     # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#     text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=150, length_function=len)
#     docs = text_splitter.split_documents(documents)
#     print("\n\n line 120", docs)
#     chunks = text_splitter.split_documents(documents)
#     print("\n\n line 118", chunks)
#     return chunks

# def store_embeddings_in_pinecone(chunks):
#     """Stores document embeddings in Pinecone."""
#     if chunks:
#         PineconeVectorStore.from_documents(chunks, embeddings, index_name=PINECONE_INDEX_NAME)
#         print(f"Embeddings for {len(chunks)} chunks stored in Pinecone index '{PINECONE_INDEX_NAME}'.")
#     else:
#         print("No document chunks to store in Pinecone.")

# if __name__ == "__main__":
#     pdf_folder = "./Earnings Call Transcripts"  # Replace with the actual path to your folder
#     if not os.path.exists(pdf_folder):
#         os.makedirs(pdf_folder)
#         print(f"Created folder '{pdf_folder}'. Please place your PDF files inside.")
#     else:
#         document_chunks = load_and_process_pdfs(pdf_folder)
#         store_embeddings_in_pinecone(document_chunks)

# ===========================================


import os
import re
import asyncio  # For parallel processing
import concurrent.futures  # For parallel processing
from typing import List, Tuple, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec, SparseValues, Vector
from PyPDF2 import PdfReader
import logging
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
from langchain_pinecone import PineconeVectorStore


 # --- Logging Setup ---
logging.basicConfig(
  level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
 )

 # --- API Keys and Initialization ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "salesforce-ear"

embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(
  temperature=0.1, model_name="gpt-4.1", openai_api_key=OPENAI_API_KEY
 )

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
des = pc.describe_index(PINECONE_INDEX_NAME)
print("\n\n line 93", des)
host = des['host']
print("\n\n line 99", pc.list_indexes())
index = pc.Index(host=host)

 # --- Helper Functions ---
def extract_metadata_from_filename(filename: str) -> Dict:
  """
  Extracts metadata from the filename using regular expressions.
  Adapt this function to your specific naming conventions.
  """

  metadata = {
   "source": filename,
   "company": "Unknown",
   "call_type": "Unknown",
   "fiscal_year": "Unknown",
   "fiscal_quarter": "Unknown",
   "call_date": "Unknown",
   "document_date": "Unknown",
   "total_pages": "Unknown",
   "section": "Unknown",
   "speaker": "Unknown",
   "topic": "General",
   "entities": [],
   "summary": "Unknown",
  }

  # --- Example: Extracting from filename patterns ---
  match = re.search(r"(\w+)_Q(\d)FY(\d{2})", filename, re.IGNORECASE)
  if match:
   metadata["company"] = match.group(1)
   metadata["fiscal_quarter"] = f"Q{match.group(2)}"
   metadata["fiscal_year"] = f"FY{match.group(3)}"

  match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)  # YYYY-MM-DD
  if match:
   metadata["call_date"] = match.group(1)
   metadata["document_date"] = match.group(1)

  return metadata

def chunk_with_metadata_from_pdf(
  pdf_path: str, splitter: RecursiveCharacterTextSplitter, base_metadata: Dict, llm: ChatOpenAI
 ) -> Tuple[List[Dict], int]:
  """
  Chunks text from a PDF, adds detailed metadata using LLM, and calculates embeddings.
  """

  reader = PdfReader(pdf_path)
  chunk_data = []
  for page_num, page in enumerate(reader.pages):
   text = page.extract_text()
   if text:
    chunks = splitter.split_text(text)
    for i, chunk in enumerate(chunks):
     chunk_id = f"{base_metadata['source']}_page_{page_num+1}_chunk_{i}"
     chunk_metadata = base_metadata.copy()
     chunk_metadata["page"] = page_num + 1
     chunk_metadata["chunk_num"] = i
     chunk_metadata["text"] = chunk  # Keep text for LLM metadata extraction

     # --- LLM Metadata Extraction ---
     try:
      llm_metadata = extract_chunk_metadata_with_llm(llm, chunk)
      chunk_metadata.update(llm_metadata)  # Merge LLM-extracted metadata
     except Exception as e:
      logging.warning(f"LLM Metadata Extraction Failed for chunk: {chunk_id} - {e}")

     chunk_data.append(
      {
       "id": chunk_id,
       "values": [],  # Embeddings added later
       "metadata": chunk_metadata,
      }
     )
   else:
    logging.warning(f"No text extracted from page {page_num+1} in {pdf_path}")
  return chunk_data, len(reader.pages)

def extract_chunk_metadata_with_llm(llm: ChatOpenAI, chunk_text: str) -> Dict:
  """
  Extracts detailed metadata from a text chunk using an LLM.
  """

  prompt = f"""
  You are an expert at extracting detailed metadata from text chunks.
  Analyze the following text and extract the following metadata.
  If information is not present, use "Unknown".

  -   section: (The section of the document this chunk belongs to. e.g., 'Introduction', 'Q&A', 'Financials')
  -   speaker: (The primary speaker in this chunk, if any. e.g., 'Marc Benioff', 'Analyst')
  -   topic: (The main topic discussed in this chunk. e.g., 'Revenue Growth', 'AI Strategy', 'Competition')
  -   entities: (A list of named entities mentioned in this chunk. e.g., ['Salesforce', 'Einstein', 'AWS'])
  -   summary: (A concise summary of the chunk in 1-2 sentences)

  Text:
  {chunk_text}

  Metadata:
  """

  llm_output = llm.invoke(prompt)  # ✅ use `invoke`, not `call_as_list`
  return parse_llm_metadata(llm_output)

  # --- Parse LLM Output ---
  metadata = parse_llm_metadata(llm_output)  # Implement this function!
  return metadata

def parse_llm_metadata(llm_output: str) -> Dict:
  """
  Parses the LLM's text output into a dictionary.
  This is a crucial step, and you'll need to adapt it.
  """
  metadata = {}
  lines = llm_output.strip().split("\n")
  for line in lines:
   if ":" in line:
    key, value = line.split(":", 1)
    key = key.strip().lower()
    value = value.strip()
    if key == "entities":
     value = (
      eval(value) if "[" in value and "]" in value else []
     )  # Safely parse list
    metadata[key] = value
  return metadata

def create_sparse_vector(text: str) -> Tuple[List[int], List[float]]:
    words = re.findall(r"\b\w+\b", text.lower())
    word_to_index = {word: idx for idx, word in enumerate(set(words))}

    indices = []
    values = []

    for word, idx in word_to_index.items():
        indices.append(idx)
        values.append(float(words.count(word)))

    return indices, values



# def upsert_data_with_hybrid_vectors(
#   index: Pinecone.Index, data: List[Dict]
#  ) -> None:
#   """
#   Upserts data to Pinecone with both dense (embeddings) and sparse vectors.
#   """
#   vectors_to_upsert = []
#   for item in data:
#    dense_vector = embeddings_model.embed_query(item["text"])
#    dense_vector_float = [float(x) for x in dense_vector] 
#    # OR, more efficiently using numpy:
#    # dense_vector_float = np.array(dense_vector, dtype=np.float32).tolist()

#    sparse_terms, sparse_counts = create_sparse_vector(item["text"])

#    sparse_vector = SparseValues(indices=sparse_terms, values=sparse_counts)

#    vectors_to_upsert.append(
#     Vector(
#      id=item["id"],
#      values=dense_vector_float,  # Use the float version
#      metadata=item["metadata"],
#      sparse_values=sparse_vector,
#     )
#    )
#   index.upsert(vectors=vectors_to_upsert)


def upsert_data_with_hybrid_vectors(index: Pinecone.Index, data: List[Dict], batch_size: int = 30):
    """
    Upserts data to Pinecone with dense + sparse vectors in smaller batches to avoid size limits.
    """
    from math import ceil

    texts = [item["text"] for item in data]
    dense_vectors = embeddings_model.embed_documents(texts)

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        vectors_to_upsert = []

        for j, item in enumerate(batch):
            idx = i + j
            sparse_terms, sparse_counts = create_sparse_vector(item["text"])
            sparse_vector = SparseValues(
                indices=sparse_terms,
                values=[float(x) for x in sparse_counts]
            )
            vectors_to_upsert.append(
                Vector(
                    id=item["id"],
                    values=dense_vectors[idx],
                    metadata=item["metadata"],
                    sparse_values=sparse_vector,
                )
            )
        # Upsert batch
        index.upsert(vectors=vectors_to_upsert)



 # --- Main Processing ---
async def process_pdf(pdf_path: str, text_splitter: RecursiveCharacterTextSplitter, llm: ChatOpenAI) -> List[Dict]:
  """
  Processes a single PDF file, extracts metadata, chunks it, and returns the chunked data.
  """
  filename = os.path.basename(pdf_path)
  try:
   # 1. Extract Metadata from Filename
   base_metadata = extract_metadata_from_filename(filename)

   # 2. Chunk with LLM Metadata Extraction
   chunked_data, total_pages = chunk_with_metadata_from_pdf(
    pdf_path, text_splitter, base_metadata, llm
   )
   base_metadata["total_pages"] = total_pages

   # 3. Add base metadata to each chunk
   for item in chunked_data:
    item["metadata"].update(base_metadata)
    item["text"] = item["metadata"].pop("text")  # Separate text for hybrid search

   logging.info(f"Processed: {filename}")
   return chunked_data

  except Exception as e:
   logging.error(f"Error processing {filename}: {e}")
   return []

async def main():
  pdf_folder = "./Earnings Call Transcripts"  # **CHANGE THIS TO YOUR ACTUAL FOLDER PATH**
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

  pdf_files = [
   os.path.join(pdf_folder, filename)
   for filename in os.listdir(pdf_folder)
   if filename.lower().endswith(".pdf")
  ]

  # --- Parallel Processing ---
  with concurrent.futures.ThreadPoolExecutor(
   max_workers=os.cpu_count()
  ) as executor:  # Use ThreadPoolExecutor for I/O-bound tasks
   loop = asyncio.get_event_loop()
   tasks = [
    await loop.run_in_executor(
     executor, process_pdf, pdf_file, text_splitter, llm
    )  # Run CPU-intensive tasks in executor
    for pdf_file in pdf_files
   ]
   all_chunked_data = await asyncio.gather(*tasks)

  # --- Flatten and Upsert ---
  all_chunked_data_flat = [
   item for sublist in all_chunked_data if sublist for item in sublist
  ]  # Flatten list of lists
  if all_chunked_data_flat:
   upsert_data_with_hybrid_vectors(index, all_chunked_data_flat)
   logging.info("All files processed and upserted.")
  else:
   logging.warning("No PDF files found in the folder.")

if __name__ == "__main__":
  asyncio.run(main())