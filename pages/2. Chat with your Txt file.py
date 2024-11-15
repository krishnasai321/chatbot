import streamlit as st
import openai
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings

# Custom title for the page
st.set_page_config(page_title="Chat with your Txt file", page_icon="📝")

# Title of the app
st.title("📝 File Q&A with Langchain and OpenAI - Vector index and similarity search based")

save_folder = "uploaded_pdfs"
os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist

# File uploader and question input
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

# Check if API key is provided
openai_api_key = st.secrets["OPENAI_API_KEY"]

def replace_t_with_space(list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document.

    Args:
        list_of_documents: A list of document objects, each with a 'page_content' attribute.

    Returns:
        The modified list of documents with tab characters replaced by spaces.
    """

    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')  # Replace tabs with spaces
    return list_of_documents
    
def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded book content.
    """

    # Load PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore

# Langchain setup
if uploaded_file and question and openai_api_key:
    
    save_path = os.path.join(save_folder, uploaded_file.name)
    # Write the file to the specified folder
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Read the uploaded file content
    chunks_vector_store = encode_pdf(save_path, chunk_size=100, chunk_overlap=20)

    # Set the OpenAI API key for Langchain
    openai.api_key = openai_api_key

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

    retriever = chunks_vector_store.as_retriever()

    # Set up system prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        
    ])

    # Create the question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Run the chain to generate an answer
    answer= rag_chain.invoke({"input": "What is the main skillset of the candidate?"})
    final_answer = answer['answer']

    # Display the result
    st.write("### Answer")
    st.write(final_answer)

