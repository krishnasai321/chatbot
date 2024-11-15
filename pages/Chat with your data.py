import streamlit as st
import openai
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


# Custom title for the page
st.set_page_config(page_title="Chat with your data", page_icon="📝")

# Title of the app
st.title("📝 Data Q&A with Langchain and OpenAI")

# File uploader and question input
uploaded_file = st.file_uploader("Upload your dataset", type=("csv"))
question = st.text_input(
    "Ask something about the dataset",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

# Check if API key is provided
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Langchain setup
if uploaded_file and question and openai_api_key:
    df = pd.read_csv(uploaded_file)

    # Set the OpenAI API key for Langchain
    openai.api_key = openai_api_key

    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True,allow_dangerous_code=True)

    answer = agent.invoke(question).output

    # Display the result
    st.write("### Answer")
    st.write(answer)

