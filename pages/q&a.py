import streamlit as st
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Title of the app
st.title("📝 File Q&A with Langchain and OpenAI")

# File uploader and question input
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

# Check if API key is provided
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Show info message if API key is missing
if uploaded_file and question and not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")

# Langchain setup
if uploaded_file and question and openai_api_key:
    # Read the uploaded file content
    article = uploaded_file.read().decode()

    # Set the OpenAI API key for Langchain
    openai.api_key = openai_api_key

    # Create a text loader for the uploaded article
    # TextLoader is used to load and parse plain text or markdown
    text_loader = TextLoader(text=article)

    # Vectorize the document (convert text into embeddings for similarity search)
    embeddings = OpenAIEmbeddings()

    # Use FAISS to store the embeddings
    vectorstore = FAISS.from_documents([article], embeddings)

    # Create a retrieval-based chain using Langchain
    # The `QA` chain is a good starting point for a simple document-based Q&A
    llm = OpenAI(temperature=0.7)

    # Prompt template for asking questions
    prompt_template = """You are a helpful assistant. Please answer the following question based on the article provided.

    Article:
    {article}

    Question: {question}

    Answer:"""

    # Create the prompt template
    prompt = PromptTemplate(input_variables=["article", "question"], template=prompt_template)

    # Create the LLMChain using OpenAI's GPT
    chain = LLMChain(llm=llm, prompt=prompt)

    # Pass the article and question into the chain for an answer
    answer = chain.run(article=article, question=question)

    # Display the result
    st.write("### Answer")
    st.write(answer)
