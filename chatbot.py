#READING AND SAVING IT IN A FILE, IN REALTIME WE USE ACTAUL VECTOR DATABASE LIKE PINECONE 

import os
import streamlit as st
from langchain_openai import OpenAI as OpenAILLM
from langchain_openai import OpenAIEmbeddings as LLMEmbeddings # Updated import
from langchain.chains import RetrievalQAWithSourcesChain as QAChain
from langchain.text_splitter import RecursiveCharacterTextSplitter as TextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader as URLDataLoader
from langchain_community.vectorstores import FAISS as FaissIndex
from dotenv import load_dotenv
import time

def load_api_key(auth_file):
    with open(auth_file, 'r') as file:
        for line in file:
            if line.startswith('OPENAI_API_KEY'):
                return line.strip().split('=')[1]
    return None

# Load OpenAI API key
openai_api_key = load_api_key('.auth')

# Check if the API key is loaded
if openai_api_key is None:
    raise ValueError("OpenAI API key not found in .auth file")


st.set_page_config(page_title="Intellitax Research Tool", layout="wide")

def set_background_color():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #76ABAE;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
set_background_color()

# Load environment variables
load_dotenv()

st.title("🔍 Intellitax Research Tool")

st.sidebar.header("🔗 Enter URLs Here")
input_urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
start_processing = st.sidebar.button("Submit")

#Placeholder for real-time status updates
status_container = st.container()

faiss_directory = "faiss_store"

# Initialize language model with set parameters
language_model = OpenAILLM(api_key=openai_api_key, temperature=0.9, max_tokens=500)
llm_embeddings = LLMEmbeddings(api_key=openai_api_key)

expected_minimum_chunks = 10

def split_text(loaded_data):
    # Primary delimiters
    primary_delimiters = ['\n\n', '\n', '.', ',']
    # Fallback delimiters
    fallback_delimiters = [';', ' ', '|']

    doc_splitter = TextSplitter(separators=primary_delimiters, chunk_size=1000)
    split_documents = doc_splitter.split_documents(loaded_data)

    # If splitting fails or returns too few segments, try fallback delimiters
    if not split_documents or len(split_documents) < expected_minimum_chunks:
        doc_splitter = TextSplitter(separators=fallback_delimiters, chunk_size=1000)
        split_documents = doc_splitter.split_documents(loaded_data)

    return split_documents

if start_processing:
    status_container.info("Processing URLs...")
    # Load and process data from provided URLs
    try:
        # Load and process data from provided URLs
        url_loader = URLDataLoader(urls=input_urls)
        loaded_data = url_loader.load()

        # Modify Text Splitter as per your data structure
        # doc_splitter = TextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
        split_documents = split_text(loaded_data)

        # Creating embeddings and FAISS index
        
        vectorindex_openai = FaissIndex.from_documents(split_documents, llm_embeddings)

        # Save FAISS index
        vectorindex_openai.save_local("faiss_store")
        status_container.success("URLs processed successfully!")
    except Exception as e:
        status_container.error(f"Error processing URLs: {e}")

user_query = st.text_input("🔍 Type your query here")

if user_query:
    # Check if the FAISS index exists and load it
    if os.path.isdir(faiss_directory):
        loaded_faiss_index = FaissIndex.load_local(faiss_directory, llm_embeddings, allow_dangerous_deserialization=True)

        # Setting up the QA Chain with the loaded index
        qa_chain = QAChain.from_llm(llm=language_model, retriever=loaded_faiss_index.as_retriever())

        # Retrieving the answer to the user's query
        query_result = qa_chain({"question": user_query}, return_only_outputs=True)

        # Displaying answer and sources
        st.subheader("Answer")
        st.write(query_result["answer"])

        # Extracting and displaying sources, if available
        result_sources = query_result.get("sources", "")
        if result_sources:
            st.subheader("Sources")
            st.write(result_sources.replace("\n", ", "))
    else:
        status_container.error("FAISS index not found. Please process the URLs first.") this is the code now give Documentation explaining the model architecture, approach to retrieval, and how
generative responses are created.
