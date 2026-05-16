import streamlit as st
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

st.title("RAG Chatbot")



@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

@st.cache_resource
def create_index():

    pdf_path = "pdfs/sample.pdf"

    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    embeddings = embed_model.encode(
        chunks,
        batch_size=8
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index, chunks

index, chunks = create_index()



query = st.text_input("Ask a question")

if query:

    

    query_embedding = embed_model.encode([query])

    

    k = 3

    distances, indices = index.search(
        np.array(query_embedding),
        k
    )

    

    retrieved_chunks = []

    for i in indices[0]:
        retrieved_chunks.append(chunks[i])

    context = "\n".join(retrieved_chunks)

    prompt = f"""
    Answer using only this context:

    {context}

    Question:
    {query}
    """

    

    response = ollama.chat(
        model="tinyllama",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    

    st.subheader("Answer")

    st.write(response["message"]["content"])

    st.subheader("Retrieved Chunks")

    for chunk in retrieved_chunks:
        st.write(chunk)
        st.write("------")