# in the name of Allah, the merciful, the beneficient, peace be upon the prophet and his progeny
# make a basic ui to take user input and display the result
import streamlit as st
from backend import crawl_links, create_vector_db, qa_retreival_chain
from urls import urls

# app's title
st.title('Document Summarizer')

# user's input question
st.write('')
st.write('')
st.write('')
question = st.text_input(label='**What question do you have?**', label_visibility='visible')

# question type
st.write('')
st.write('')
st.write('')
type_question = st.selectbox('What type does the question fall under?', ('Maths','Medical','Tech','Physics','Chemistry','Biology','History','Psychology','Religion','Geography','Economics','News','Fashion','Art','Movies','Other'),
                             placeholder='Select question Type ...')

st.write('')
st.write('')

# fetch the urls related to the genre
urls_list = urls[type_question]
print(urls_list)

# load the docs from the web
pages = crawl_links(urls_list)

# create the vector store
vector_store = create_vector_db(pages)

# get the summary
summary = qa_retreival_chain(question, vector_store)

# answer to be displayed
st.write('**Answer:**')
expander = st.expander('See Answer Below')
expander.write(summary)

# links of the documents to be displayed
# st.markdown(
#     f'<a href={} target="_blank">OpenAI Website</a>',
#     unsafe_allow_html=True
# )
