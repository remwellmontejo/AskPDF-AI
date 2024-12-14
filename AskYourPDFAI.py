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
    st.set_page_config(page_title="AskYourPDFAI")

    if "question" not in st.session_state:
        st.session_state.question = ""

    with sidebar:
        st.header("Upload PDF files")
        uploaded_file = st.file_uploader("Upload files to start.",type="pdf", accept_multiple_files=True)
        if uploaded_file:
            with st.spinner("Reading your document..."):
                text = extract_text_from_pdf(uploaded_file)
                vector_store = create_faiss_vector_store(text)
                qa_chain = build_qa_chain(vector_store)
        st.divider()
        st.caption(
            """
               **A project in collaboration with the Adolescent Health and Development (AHD) Department of Muntinlupa, initiated by the students of Colegio de Muntinlupa from CPE3B - Cognate Elective 1 (COEN3103)**
            """
        )
        st.caption(
            """
                Montejo, Shane Remwell\n
                Nillasca, Justin\n
                Mayor, Miguel\n
                Medina, Emmanuel
            """
        )


        

    st.title("Ask your PDF AI using Google Gemini")
    st.write("Upload PDF files and ask questions based on its content. Summarize data from uploaded files in seconds.")
    st.divider()

    if 'qa_chain' in locals():
        st.session_state.question = st.chat_input("Ask a question about the uploaded PDF")
        if st.session_state.question:
            st.subheader("Prompt:")
            st.write(st.session_state.question)
            st.divider()
            answer = qa_chain.run(st.session_state.question)
            st.subheader("Response:")
            st.write_stream(stream_data(answer))

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

if __name__ == "__main__":
    main()
