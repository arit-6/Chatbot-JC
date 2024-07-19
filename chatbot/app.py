import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
import time

load_dotenv()

# Load the Groq and Google API key from the environment variables   
groq_key = os.environ.get("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')


st.title("ChatBOT")

llm = ChatGroq(api_key=groq_key, model="mixtral-8x7b-32768")

# Use ChatMessagePromptTemplate for prompt definition (fixed)

ini_prompt = """
Answer the question based on the provided context only.
Please provide the most accurate responses for each question in a detailed manner,
when answering questions please assume its your role to answer the question,
please keep in mind this and dont
include any preamble or introduction text in your response. And plesse dont include 'Based on the provided 
context sentence.' in your response.
If the answer is not in the context then say contact the business, dont makeup an answer.
context: {context}
Questions:
{input}
"""

prompt = PromptTemplate(input_variables=["context", "input"], template=ini_prompt)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Load the PDF using PyPDFLoader (modified)
        st.session_state.loader = PyPDFLoader("chatbot/Data/Corpus.pdf")  ## Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  ## Document Loading (single doc)
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)


vector_embedding()

prompt1 = st.chat_input("How can I help you?")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

historical_context = "\n".join([message["content"] for message in st.session_state.messages])

if prompt1:
    retriever = st.session_state.vectors.as_retriever()
    retrived_data = retriever.get_relevant_documents(prompt1)
    context = "\n".join([item.page_content for item in retrived_data])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    start = time.process_time()
    response = llm_chain.run(input=prompt1, context=context+historical_context) 
   

    st.chat_message('user').markdown(prompt1)
    st.session_state.messages.append({'role': 'user', 'content': prompt1})
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content': response})
    st.write(f"Time taken: {time.process_time() - start} seconds")

 
    

    
