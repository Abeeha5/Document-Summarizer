from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from googleapiclient.discovery import build
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import requests_cache
import nest_asyncio
import trafilatura
import asyncio
import streamlit as st
import httpx
import os

# to avoid problems with asyncio
nest_asyncio.apply()

# load data from .env file
load_dotenv()

# api keys 
langchain_tracing_v2 = os.getenv('LANGCHAIN_TRACING_V2')
langchain_endpoint = os.getenv('LANGCHAIN_ENDPOINT')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
open_router_api_key = os.getenv('OPENROUTER_API_KEY')
google_api_key = os.getenv('GOOGLE_SEARCH_API')
cse_id = os.getenv('CSE_ID')
os.environ['USER_AGENT'] = st.secrets['USER_AGENT']

llm = ChatOpenAI(
    openai_api_key=open_router_api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-chat-v3-0324",
    temperature=0
)

# google search
def google_search(query, api_key = google_api_key, cse_id = cse_id, num_results = 2):
    service = build("customsearch", "v1", developerKey=api_key)
    results = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
    urls = [item['link'] for item in results.get('items', [])]
    return urls

# Enable persistent caching for HTTPX
requests_cache.install_cache("web_cache", expire_after=3600)  
# expire_after in seconds → 3600 sec = 1 hour 

# fetches the url
async def fetch(session, url):
    try:
        resp = await session.get(url, timeout=10)
        if resp.status_code == 200:
            return url, resp.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return url, None

# fetch content from multiple sites in parallel
async def scrap_sites(urls):
    docs = []
    async with httpx.AsyncClient(headers={"User-Agent": "MyFastCrawler/1.0"}) as client:
        results = await asyncio.gather(*(fetch(client, url) for url in urls))
        for url, html in results:
            if html:
                text = trafilatura.extract(html)
                if text:
                    docs.append(Document(page_content=text, metadata={"source": url}))
    return docs

# split the retreived docs and store in vector db
def create_vector_db(pages):

    # print(len(pages)) 
    splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 100)

    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store

# make a retreiver
def qa_retreival_chain(question, vector_store):

    # make a vectorstore retreiver
    retriever = vector_store.as_retriever()
    
    # define the prompt template
    template = (
        'Ok so you are given the following question {question}. And this is the text you are supposed to summarize the context {context} given the question. If you do not know the answer tell that you do not know. Do not mention that you are answering from a document. Do not use I or me , tell it as if you are informing.'
    )

    # prompt
    prompt = ChatPromptTemplate.from_template(template)

    # chain
    rag_chain = ( 
            {'context': retriever, 'question': RunnablePassthrough()} |
            prompt | 
            llm | 
            StrOutputParser()
        )

    summary = rag_chain.invoke(question)

    return summary







    
    

