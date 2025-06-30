import streamlit as st
from openai import OpenAI
import json
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional

# --- Configuration ---
SYSTEM_PROMPT = """
You are Alex, a knowledgeable university assistant. Respond with:
- Accurate information from provided data
- Friendly, professional tone
- Concise answers (1-3 paragraphs)
- Markdown formatting when helpful
- "I don't know" for unclear queries (never hallucinate)
"""

# --- Initialize OpenAI Client ---
@st.cache_resource(show_spinner=False)
def init_openai_client() -> tuple[OpenAI, OpenAIEmbeddings]:
    """Initialize OpenAI client and embeddings with API key validation"""
    try:
        api_key = (
            os.getenv("OPENAI_API_KEY") or 
            st.secrets.get("OPENAI_API_KEY") or
            st.session_state.get("openai_api_key")
        )
        
        if not api_key:
            st.error("🔑 API key missing. Please add OPENAI_API_KEY to secrets.toml or enter below")
            st.stop()
            
        return OpenAI(api_key=api_key), OpenAIEmbeddings(openai_api_key=api_key)
    except Exception as e:
        st.error(f"🚨 OpenAI initialization failed: {str(e)}")
        st.stop()

client, embeddings = init_openai_client()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# --- Data Loading ---
@st.cache_data(show_spinner="Loading university knowledge...")
def load_university_data() -> Dict:
    """Load structured Q&A data with validation"""
    try:
        with open("qa_data.json") as f:
            data = json.load(f)
            
            # Validate structure
            if not isinstance(data, dict) or "faqs" not in data:
                st.error("❌ Invalid qa_data.json format. Needs 'faqs' array")
                st.stop()
                
            return data
    except FileNotFoundError:
        st.warning("ℹ️ No qa_data.json found - using empty knowledge base")
        return {"faqs": []}
    except Exception as e:
        st.error(f"📂 Error loading qa_data.json: {str(e)}")
        st.stop()

university_data = load_university_data()

# --- Session Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm Alex, your university assistant. Ask me about:\n\n"
         "- Admissions\n- Courses\n- Campus life\n- And more!"}
    ]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Document Processing ---
def process_documents(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> Optional[FAISS]:
    """Process uploaded documents into searchable vector store"""
    docs = []
    
    for file in files:
        try:
            # Save temp file with proper extension
            ext = os.path.splitext(file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(file.getbuffer())
                tmp_path = tmp.name
            
            # Select appropriate loader
            loader_class = {
                ".pdf": PyPDFLoader,
                ".docx": Docx2txtLoader,
                ".txt": TextLoader
            }.get(ext, TextLoader)
            
            docs.extend(loader_class(tmp_path).load())
            os.unlink(tmp_path)
        except Exception as e:
            st.toast(f"⚠️ Skipped {file.name}: {str(e)}", icon="⚠️")
    
    if docs:
        split_docs = text_splitter.split_documents(docs)
        return FAISS.from_documents(split_docs, embeddings)
    return None

# --- Response Generation ---
def generate_response(prompt: str) -> str:
    """Generate context-aware response using multiple knowledge sources"""
    try:
        # 1. Check exact FAQ matches first
        lower_prompt = prompt.lower()
        for faq in university_data.get("faqs", []):
            if (lower_prompt in faq["question"].lower() or 
                faq["question"].lower() in lower_prompt):
                return f"{faq['answer']}\n\n*(Source: University FAQs)*"
        
        # 2. Build context from multiple sources
        context_parts = [
            f"# University Knowledge Base\n{json.dumps(university_data, indent=2)}"
        ]
        
        # Add document context if available
        if st.session_state.vector_store:
            docs = st.session_state.vector_store.similarity_search(prompt, k=3)
            context_parts.append("# Supplemental Documents\n" + 
                               "\n---\n".join(d.page_content for d in docs))
        
        full_context = "\n\n".join(context_parts)
        
        # 3. Generate LLM response
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",  # Updated model
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{full_context}\n\nQuestion: {prompt}"}
            ],
            temperature=0.5,  # More deterministic
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.toast(f"⚠️ Generation error: {str(e)}", icon="⚠️")
        return "Sorry, I encountered an error. Please try again."

# --- UI Components ---
st.set_page_config(page_title="University Assistant", page_icon="🎓")
st.title("🎓 University Assistant")
st.caption("Ask me about admissions, courses, campus services, and more!")

with st.sidebar:
    st.header("Configuration")
    
    # API key input (only if not in secrets)
    if not os.getenv("OPENAI_API_KEY"):
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Get yours from platform.openai.com"
        )
    
    # Document upload section
    st.header("Knowledge Expansion")
    uploaded_files = st.file_uploader(
        "Upload university documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Add handbooks, policies, or course catalogs"
    )
    
    if uploaded_files and st.button("Process Documents", type="primary"):
        with st.spinner("Analyzing documents..."):
            st.session_state.vector_store = process_documents(uploaded_files)
            if st.session_state.vector_store:
                st.toast(f"Processed {len(uploaded_files)} documents!", icon="✅")
            else:
                st.toast("No valid documents processed", icon="⚠️")

# --- Chat Interface ---
for msg in st.session_state.messages:
    avatar = "🎓" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the university..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="🎓"):
        response = generate_response(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
