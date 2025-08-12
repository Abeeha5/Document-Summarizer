# WEB-BASED DOCUMENT SUMMARIZER

This is a simple document summarizer which works as follows:

1. The user enters the question.
2. The question is given to google's custom search api which then finds relevant pages across the web.
3. The async httpx is used to fetch the pages.
4. The trafilatura library is used to extract the useful content from the top 2 pages.
5. The retreived docs are then split and their embeddings, in this case the HuggingFace all-MiniLM-L6-v2 model embeddings, are stored in the a vector store namely FAISS.
6. The context and the question are fed to the deepseek's deepseek-chat-v3-0324 model which then summarizes and presents the final result.
7. The links to the relavant pages from the web are also provided
   
