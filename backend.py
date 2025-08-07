from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.base import BaseLoader
from langchain.agents import initialize_agent, Tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv
import asyncio
import os
import sys

# to avoid problems with asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# load data from .env file
load_dotenv()

# api keys 
langchain_tracing_v2 = os.getenv('LANGCHAIN_TRACING_V2')
langchain_endpoint = os.getenv('LANGCHAIN_ENDPOINT')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
open_router_api_key = os.getenv('OPEN_ROUTER_API_KEY')

llm = ChatOpenAI(
    openai_api_key=open_router_api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    model="openrouter/horizon-beta",
    temperature=0
)

# set up a langchain tool using crawl4ai
class Crawl4AILoader(BaseLoader):

    def __init__(self, urls: str, browser_config=None, run_config=None):
        self.urls = urls
        self.browser_config = browser_config
        self.run_config = run_config
    
    async def load(self) -> list[Document]:
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            await crawler.start()
            res = await crawler.arun(self.urls, config=self.run_config)
        doc = Document(page_content = res.markdown, metadata = {'url': res.url})
        return [doc]

# wrap it as a tool in the langchain agent
tool = Tool(
    name = 'crawl_web',
    func = lambda url: asyncio.run(Crawl4AILoader(url).load()),
    description = 'Crawl a url and return text content'
)

agent = initialize_agent([tool], llm = llm, agent_type="zero-shot-react-description", verbose=True)

# web crawl to fetch docs using firecrawl api
def crawl_links(urls):
    pages = []
    for url in urls:
        docs = asyncio.run(Crawl4AILoader(url).load())
        pages.extend(docs)
    return pages    

# split the retreived docs and store in vector db
def create_vector_db(pages):

    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)

    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store

# make a retreiver
def qa_retreival_chain(question, vector_store):

    # make a vectorstore retreiver
    retriever = vector_store.as_retriever()

    # specify the llm to be used 
    llm = ChatOpenAI(
            model='openrouter/horizon-beta',
            temperature=0 )
    
    # retreive the relevant docs
    # context = retreiver.get_relevant_documents(question)
    
    # define the prompt template
    template = (
        'Ok so you are given the following question {question}. And this is the text you are supposed to summarize the context {context} given the question. If you do not know the answer tell that you do not know.'
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







    
    

