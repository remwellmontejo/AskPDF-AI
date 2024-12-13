import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from streamlit import sidebar

genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY"))

def extract_text_from_pdf(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def create_faiss_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

def build_qa_chain(vector_store):
    retriever = vector_store.as_retriever()
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest', temperature=0.5)
    qa_chain = load_qa_chain(llm=llm, chain_type="stuff")
    qa_chain = RetrievalQA(retriever=retriever,combine_documents_chain=qa_chain)
    return qa_chain

def main():
    with sidebar:
        st.title('Ask your PDF AI')
        st.subheader('A project for Cognate Elective 3103')
        st.text(
            """
                Montejo, Shane Remwell C.
                Nillasca, Justin
                Mayor, Miguel
                Medina, Emmanuel
            """
        )
        st.divider()

        uploaded_file = st.file_uploader("Upload files to start.",type="pdf", accept_multiple_files=True)
        if uploaded_file:
            with st.spinner("Reading your document..."):
                text = extract_text_from_pdf(uploaded_file)
                vector_store = create_faiss_vector_store(text)
                qa_chain = build_qa_chain(vector_store)

    st.title("Ask your PDF AI")
    st.write("Upload a PDF and ask questions based on its content.")
    st.divider()

    if 'qa_chain' in locals():
        question = st.chat_input("Ask a question about the uploaded PDF")
        if question:
            answer = qa_chain.run(question)
            st.write_stream(stream_data(answer))

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

if __name__ == "__main__":
    main()