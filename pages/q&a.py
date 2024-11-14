import streamlit as st
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

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

    # Create a prompt template
    prompt_template = """You are a helpful assistant. Below is an article and a question. Please provide a detailed answer based on the article.

    Article:
    {article}

    Question: {question}

    Answer:"""

    # Create the prompt template for Langchain
    prompt = PromptTemplate(input_variables=["article", "question"], template=prompt_template)

    # Set up the OpenAI model to GPT-3.5 Turbo
    llm = OpenAI(temperature=0.7, model="gpt-3.5-turbo")

    # Create the LLMChain with Langchain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain to generate an answer
    answer = chain.run(article=article, question=question)

    # Display the result
    st.write("### Answer")
    st.write(answer)
