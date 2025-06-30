import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ✅ List your PDF file paths here
pdf_paths = [
    r"data/medical_book1.pdf",
    r"data/medical_book2.pdf"
]

all_docs = []

# ✅ Load and split documents
for pdf_path in pdf_paths:
    if os.path.exists(pdf_path):
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        all_docs.extend(pages)
    else:
        print(f"❌ File not found: {pdf_path}")

# ✅ Split text
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(all_docs)

# ✅ Embedding and FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(split_docs, embedding_model)

# ✅ Save vector index locally
db.save_local("index")

print("✅ FAISS index created and saved.")
