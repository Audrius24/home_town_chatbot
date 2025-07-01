import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import bs4

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("SECRET")

# Load documents from 3 sources (1 file + 2 websites)
file_loader = TextLoader(os.path.abspath("plunge.txt"))

with open("plunge.txt", encoding="utf-8") as f:
    print(f.read())

wiki_loader = WebBaseLoader(
    web_paths=["https://en.wikipedia.org/wiki/Plung%C4%97"],
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="bodyContent"))
)

travel_loader = WebBaseLoader(
    web_paths=["https://www.lithuania.travel/en/place/plunge-manor"],
    bs_kwargs=dict(parse_only=bs4.SoupStrainer("main"))
)

docs = file_loader.load() + wiki_loader.load() + travel_loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# Vector store using Chroma
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key
    ),
)

retriever = vectorstore.as_retriever()

# Prompt template from LangChain Hub
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Streamlit UI
st.title("📍 Plungė RAG Chatbot")

def generate_response_with_sources(input_text):
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        api_key=openai_api_key
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(input_text)
    st.success(response)

    with st.expander("📚 View retrieved sources"):
        relevant_docs = retriever.get_relevant_documents(input_text)
        for i, doc in enumerate(relevant_docs):
            st.markdown(f"**Chunk {i+1}:**\n\n{doc.page_content[:500]}...\n")

# Input form
with st.form("input_form"):
    user_input = st.text_area("Ask something about Plungė:", "What is Plungė known for?")
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response_with_sources(user_input)
