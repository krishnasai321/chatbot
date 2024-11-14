import streamlit as st
import openai

# Streamlit main title and file uploader
st.title("📝 File Q&A with OpenAI GPT")
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

# Check if API key is provided
openai_api_key = st.secrets["OPENAI_API_KEY"]

if uploaded_file and question and not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")

if uploaded_file and question and openai_api_key:
    article = uploaded_file.read().decode()
    
    # Define the prompt for GPT model
    prompt = f"""
    You are a helpful assistant. Below is an article and a question. Please provide a detailed answer based on the article.

    Article:
    {article}

    Question: {question}
    """

    # Set the OpenAI API key
    openai.api_key = openai_api_key

    # Use the latest completion endpoint
    response = openai.Completion.create(
        model="gpt-3.5-turbo",  # or "gpt-3.5-turbo" for cheaper option
        prompt=prompt,
        max_tokens=100,  # Adjust tokens if necessary
        temperature=0.7,  # Adjust for creativity (0.0 to 1.0)
    )
    
    # Display the model's response
    st.write("### Answer")
    st.write(response.choices[0].text.strip())
