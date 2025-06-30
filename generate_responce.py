import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables (GOOGLE_API_KEY)
load_dotenv()

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS vectorstore
try:
    vectorstore = FAISS.load_local(
        "index",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True  # ✅ only if YOU created the index
    )
except Exception as e:
    print("❌ Failed to load FAISS index:", e)
    vectorstore = None

# Load Gemini LLM (Flash version)
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.4
)

# Set up retrieval-based chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever() if vectorstore else None,
    return_source_documents=False
) if vectorstore else None

# Core function for getting response
def get_response(query, chat_history=[]):
    if not qa_chain:
        return {"answer": "⚠️ Vector index not available."}
    try:
        print("📩 Query:", query)

        # Prompt Gemini to reply concisely and focused
        prompt = f"Answer briefly (3-4 lines only) and directly: {query}"

        result = qa_chain.invoke({
            "question": prompt,
            "chat_history": chat_history
        })
        print("✅ Gemini Answer:", result)

        if isinstance(result, dict):
            return {"answer": result.get("answer") or result.get("result") or str(result)}
        else:
            return {"answer": str(result)}

    except Exception as e:
        print("❌ Error in get_response:", e)
        return {"answer": f"❌ Error: {e}"}
