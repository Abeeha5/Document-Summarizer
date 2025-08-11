# in the name of Allah, the merciful, the beneficient, peace be upon the prophet and his progeny

# make a basic ui to take user input and display the result
import streamlit as st
import asyncio
from backend import google_search, create_vector_db, qa_retreival_chain, scrap_sites

# app's title
st.title('Document Summarizer')

# user's input question
st.write('')
st.write('')
st.write('')
question = st.text_input(label='**What question do you have?**', label_visibility='visible')

# answer to be displayed
st.write('')
st.write('**Answer:**')
expander = st.expander('See Answer Below')

if question:

    # search for the relavant links based on question
    urls = google_search(question)

    # get the content
    docs =  asyncio.run(scrap_sites(urls))

    # create the vector store
    vector_store = create_vector_db(docs)

    # get the summary
    summary = qa_retreival_chain(question, vector_store)

    expander.write(summary)

    # links of the documents to be displayed
    st.write('')
    st.write('')
    st.write('**For more details checkout the following pages**: ')
    for url in urls:
        st.write(url)
        st.write('')
    