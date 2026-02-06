import streamlit as st
from pdf_utils import extract_text_from_pdf, chunk_text
from vector_store import VectorStore
from gemini_utils import generate_answer

st.set_page_config(page_title="PDF Chatbot (Gemini)")
st.title("PDF Chatbot with Gemini (RAG)")

uploaded_file = st.file_uploader("Upload  PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)

        vector_store = VectorStore()
        vector_store.build_index(chunks)

    st.success("PDF processed successfully!")

    question = st.text_input("Ask a question from the PDF")

    if question:
        relevant_chunks = vector_store.search(question)
        answer = generate_answer(question, relevant_chunks)

        st.subheader("Answer")
        st.write(answer)
