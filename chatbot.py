import streamlit as st
from openai import OpenAI
import json
import os
import tempfile
import time
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
SYSTEM_PROMPT = """
You are Alex, a knowledgeable university assistant. Be:
- Accurate from qa_data.json
- Warm and helpful
- Adaptive to context
- Mindful of user preferences
"""

# --- Initialize OpenAI Client ---
@st.cache_resource
def init_openai_client():
    try:
        # Try multiple ways to get API key
        api_key = (
            os.getenv("OPENAI_API_KEY") or 
            st.secrets.get("OPENAI_API_KEY") or
            st.session_state.get("openai_api_key")
        )
        
        if not api_key:
            st.error("🔑 API key missing. Please configure OPENAI_API_KEY")
            st.stop()
            
        return OpenAI(api_key=api_key), OpenAIEmbeddings(openai_api_key=api_key)
    except Exception as e:
        st.error(f"🚨 Failed to initialize OpenAI: {str(e)}")
        st.stop()

client, embeddings = init_openai_client()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

# --- Load University Data ---
@st.cache_data
def load_university_data():
    try:
        with open("qa_data.json") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"📂 Error loading qa_data.json: {str(e)}")
        st.stop()

university_data = load_university_data()

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm Alex, your university assistant. How can I help?"}
    ]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Document Processing ---
def process_documents(files):
    docs = []
    for file in files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                tmp.write(file.getbuffer())
                tmp_path = tmp.name
            
            loader = {
                ".pdf": PyPDFLoader,
                ".docx": Docx2txtLoader,
                ".txt": TextLoader
            }.get(os.path.splitext(file.name)[1].lower(), TextLoader)(tmp_path)
            
            docs.extend(loader.load())
            os.unlink(tmp_path)
        except Exception as e:
            st.sidebar.error(f"❌ Error processing {file.name}: {str(e)}")
    
    if docs:
        return FAISS.from_documents(text_splitter.split_documents(docs), embeddings)
    return None

# --- UI Components ---
st.title("🎓 University Assistant")
with st.sidebar:
    st.header("Settings")
    if not os.getenv("OPENAI_API_KEY"):
        st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password")
    
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Add supplemental files", 
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    if uploaded_files and st.button("Process"):
        with st.spinner("Indexing documents..."):
            st.session_state.vector_store = process_documents(uploaded_files)

# --- Chat Functions ---
def generate_response(prompt):
    try:
        # 1. Check direct matches
        if "faqs" in university_data:
            for faq in university_data["faqs"]:
                if prompt.lower() in faq["question"].lower():
                    return faq["answer"]
        
        # 2. Build context
        context = f"University Knowledge:\n{json.dumps(university_data, indent=2)}"
        
        if st.session_state.vector_store:
            docs = st.session_state.vector_store.similarity_search(prompt, k=2)
            context += f"\n\nSupplemental Docs:\n{docs[0].page_content if docs else 'None'}"
        
        # 3. Generate response
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# --- Chat Interface ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about the university..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        response = generate_response(prompt)
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
