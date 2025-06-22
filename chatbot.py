import streamlit as st
from openai import OpenAI
import json
import os
import tempfile
import time
from openai import OpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration
SYSTEM_PROMPT = """
You are Alex, a knowledgeable university assistant with both technical knowledge and emotional intelligence.
Combine these capabilities:
1. Answer accurately from the qa_data.json knowledge base
2. Show personality: warm, patient, and helpful
3. Adapt tone to context (formal/casual)
4. Remember user preferences and context
5. Handle both simple FAQs and complex queries
"""

# Initialize
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

# Load qa_data.json
try:
    with open("qa_data.json") as f:
        university_data = json.load(f)
except FileNotFoundError:
    st.error("qa_data.json not found. Please ensure the file exists.")
    st.stop()
except json.JSONDecodeError:
    st.error("Invalid JSON format in qa_data.json")
    st.stop()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! I'm Alex, your university assistant. How can I help you today?"}
    ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "user_info" not in st.session_state:
    st.session_state.user_info = {}

# UI Setup
st.title("🎓 University Assistant (Alex)")
st.sidebar.header("Supplemental Documents")

# File Upload Section (for additional documents)
uploaded_docs = st.sidebar.file_uploader("Upload Supporting Documents", 
                                       type=["pdf", "docx", "txt"],
                                       accept_multiple_files=True)

def process_documents(uploaded_files):
    docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        
        try:
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path)
            docs.extend(loader.load())
        except Exception as e:
            st.sidebar.error(f"Error loading {file.name}: {str(e)}")
        finally:
            os.unlink(tmp_path)
    
    if docs:
        chunks = text_splitter.split_documents(docs)
        return FAISS.from_documents(chunks, embeddings)
    return None

# Process Document Uploads
if uploaded_docs:
    with st.sidebar.status("Processing documents...", expanded=True):
        st.write("Indexing documents...")
        st.session_state.vector_store = process_documents(uploaded_docs)
        st.sidebar.success(f"Processed {len(uploaded_docs)} documents!")

# Core Chat Functions
def get_university_context():
    """Extracts structured information from qa_data.json"""
    context = "UNIVERSITY KNOWLEDGE BASE (qa_data.json):\n"
    
    # Add FAQs if they exist in the structure
    if "faqs" in university_data:
        context += "\nFAQs:\n"
        for item in university_data["faqs"]:
            context += f"Q: {item.get('question','')}\nA: {item.get('answer','')}\n"
    
    # Add other sections dynamically
    for section, data in university_data.items():
        if section != "faqs":  # Already handled
            context += f"\n{section.upper()}:\n"
            if isinstance(data, dict):
                for key, value in data.items():
                    context += f"{key}: {value}\n"
            elif isinstance(data, list):
                for item in data:
                    context += f"- {item}\n"
            else:
                context += f"{data}\n"
    
    return context

def search_vector_store(query):
    if not st.session_state.vector_store:
        return None
    docs = st.session_state.vector_store.similarity_search(query, k=2)
    return "\n".join([d.page_content for d in docs])

def analyze_emotion(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Classify emotion from text (angry/happy/confused/neutral)"},
            {"role": "user", "content": text}
        ],
        max_tokens=10
    )
    return response.choices[0].message.content.lower()

def stream_response(response):
    message_placeholder = st.empty()
    full_response = ""
    for chunk in response.split():
        full_response += chunk + " "
        time.sleep(0.05)
        message_placeholder.markdown(full_response + "▌")
    return full_response

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Alex anything about the university..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # Step 1: Check for direct matches in qa_data.json
        direct_answer = None
        if "faqs" in university_data:
            for faq in university_data["faqs"]:
                if faq["question"].lower() in prompt.lower():
                    direct_answer = faq["answer"]
                    break
        
        # Step 2: Build augmented context if needed
        if not direct_answer:
            context_parts = [get_university_context()]
            
            # Add document context if available
            if doc_context := search_vector_store(prompt):
                context_parts.append(f"SUPPLEMENTAL DOCUMENTS:\n{doc_context}")
            
            # Add user info if available
            if st.session_state.user_info:
                context_parts.append(f"USER PROFILE:\n{st.session_state.user_info}")
        
        # Step 3: Detect emotion
        emotion = analyze_emotion(prompt)
        
        # Step 4: Generate response
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *st.session_state.messages[:-1]
        ]
        
        if not direct_answer:
            messages.insert(1, {
                "role": "assistant", 
                "content": "CONTEXT:\n" + "\n\n".join(context_parts)
            })
        
        messages.append({
            "role": "user", 
            "content": f"[Detected emotion: {emotion}] {prompt}"
        })
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.7
        ).choices[0].message.content
        
        # Step 5: Stream response
        full_response = stream_response(response)
        
        # Step 6: Update user info if shared
        if "my name is" in prompt.lower():
            name = prompt.split("is")[-1].strip()
            st.session_state.user_info["name"] = name
            full_response += f"\n\nI'll remember your name is {name}!"
    
    # Add final response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
