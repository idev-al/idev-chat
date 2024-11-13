import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, Settings, Document
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import tempfile
import pymupdf  # Import PyMuPDF for PDF handling

# Set up Streamlit app configurations
st.set_page_config(page_title="Chat with the KB chat, powered by iDev", page_icon="🦙", layout="wide")
openai.api_key = st.secrets.openai_key

# Hide Streamlit branding
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stApp { overflow: hidden; padding: 0; }
        
        /* Remove padding around the content */
        .block-container {
            padding: 0 1rem;  /* Adjust the padding values as needed */
        }
        
        /* Optional: Adjust chat input padding */
        .css-1y4p8pa textarea {
            padding: 5px;  /* Adjust chat input padding */
        }
    </style>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about Sistema.bio"}]

# Google Drive API setup
def fetch_files_from_drive(folder_id):
    credentials = Credentials.from_service_account_info(st.secrets["google_service_account"])
    service = build('drive', 'v3', credentials=credentials)

    # Query for .txt and .pdf files in the specified folder
    query = f"'{folder_id}' in parents and (mimeType='text/plain' or mimeType='application/pdf')"
    results = service.files().list(q=query).execute()
    files = results.get('files', [])
    
    documents = []
    for file in files:
        request = service.files().get_media(fileId=file['id'])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        
        # Process .txt files
        if file['mimeType'] == 'text/plain':
            text_content = fh.read().decode("utf-8")
            documents.append(Document(text=text_content))
        
        # Process .pdf files using PyMuPDF
        elif file['mimeType'] == 'application/pdf':
            pdf_text = extract_text_from_pdf(fh)
            documents.append(Document(text=pdf_text))
    
    return documents


def extract_text_from_pdf(file):
    text_content = ""

    # Create a temporary file to save the BytesIO content
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
        temp_file.write(file.read())  # Write the BytesIO content to the temp file
        temp_file.flush()

        # Open the temp file with PyMuPDF
        pdf_document = pymupdf.open(temp_file.name)
        
        # Loop through each page and extract text
        for page_num in range(pdf_document.page_count):  # Loop through each page
            page = pdf_document.load_page(page_num)  # Correct method to load a page
            text_content += page.get_text("text")  # Extract text from each page
        pdf_document.close()

    return text_content

# Load data from Google Drive
@st.cache_resource(show_spinner=False)
def load_data():
    folder_id = "1z1-oqJxOgRT9NOs047FK9LO6Y89ZxksB"  # Google Drive folder ID
    docs_content = fetch_files_from_drive(folder_id)
    
    # Initialize and configure LLM
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
system_prompt="""You are an expert on Sistema.bio.
    Your primary job is to answer technical questions based on the context of the uploaded documents. Keep answers short and factual.
    However, if the answer to a question isn't found in the documents, please provide a contact us link: https://sistema.bio/contact-us/"""
    )
    
    # Create VectorStoreIndex from Document objects
    index = VectorStoreIndex.from_documents(docs_content)
    return index

index = load_data()

# Initialize chat engine
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True, streaming=True)

# Chat interaction
if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    if message["role"] == "assistant":
        # Use the URL directly for the assistant's avatar
        avatar_url = "https://sistema.bio/wp-content/uploads/2024/05/favicon-150x150.png"
    else:
        avatar_url = None  # No avatar for the user or you can set another URL

    with st.chat_message(message["role"], avatar=avatar_url):
        st.write(message["content"])

# If last message is from user, generate a response
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant", avatar="https://sistema.bio/wp-content/uploads/2024/05/favicon-150x150.png"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        response_message = {"role": "assistant", "content": response_stream.response}
        st.session_state.messages.append(response_message)
