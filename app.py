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

   import os

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

all_chunks = []

all_metadata = []

pdf_folder = "pdfs"

for file in os.listdir(pdf_folder):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(pdf_folder, file)

        doc = fitz.open(pdf_path)

        text = ""

        for page_num, page in enumerate(doc):

            page_text = page.get_text()

            chunks = splitter.split_text(page_text)

            for chunk in chunks:

                all_chunks.append(chunk)

                all_metadata.append(
                    {
                        "source": file,
                        "page": page_num + 1
                    }
                )

    embeddings = embed_model.encode(
        all_chunks,
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
        retrieved_chunks.append(all_chunks[i])

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
