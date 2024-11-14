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

# If file is uploaded, question is asked, and API key is provided
if uploaded_file and question and openai_api_key:
    article = uploaded_file.read().decode()
    
    # Define the prompt for GPT model in a conversational format
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Here is an article:\n\n{article}\n\nQuestion: {question}"}
    ]
    
    # Set the OpenAI API key
    openai.api_key = openai_api_key

    # Call OpenAI GPT model to get a response using the chat API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # You can use "gpt-3.5-turbo" if you prefer GPT-3.5
        messages=messages,
        max_tokens=100,  # You can adjust this number based on your needs
        temperature=0.7,  # Adjust temperature for response randomness
    )
    
    # Display the model's response
    st.write("### Answer")
    st.write(response['choices'][0]['message']['content'].strip())
