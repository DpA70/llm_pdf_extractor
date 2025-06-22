import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import tempfile

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "API KEY"

# Streamlit UI Setup
st.set_page_config(page_title="PDF Q&A App", layout="centered")
st.title("📄 Ask Questions from Your PDF")

# Upload the PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Ask the question
query = st.text_input("Ask a question about the uploaded PDF:")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()

    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # Create embeddings using Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    valid_texts = []
    valid_metadatas = []

    for text, meta in zip(texts, metadatas):
        try:
            vector = embeddings.embed_query(text)
            if vector:
                valid_texts.append(text)
                valid_metadatas.append(meta)
        except Exception as e:
            print(f"Skipping chunk due to embedding error: {e}")

    if not valid_texts:
        st.error("No valid text chunks could be embedded from this PDF. Please try a different file.")
        st.stop()

    # Use a temporary Chroma DB (not persistent)
    with tempfile.TemporaryDirectory() as tmp_chroma_dir:
        vectordb = Chroma.from_texts(valid_texts, embedding=embeddings,
                                     metadatas=valid_metadatas,
                                     persist_directory=tmp_chroma_dir)

        # Use Gemini Chat model
        llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.3)

        qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

        if query:
            with st.spinner("Thinking..."):
                answer = qa.run(query)
                st.markdown("### 📌 Answer:")
                st.write(answer)

    os.remove(temp_pdf_path)  # Optional: clean up uploaded file
