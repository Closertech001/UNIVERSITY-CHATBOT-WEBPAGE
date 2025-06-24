# --- Streamlit University Chatbot with All Enhancements ---
import streamlit as st
import json
import re
import time
import sqlite3
from sentence_transformers import SentenceTransformer, util
from symspellpy import SymSpell
import openai
import os
from datetime import datetime, timedelta

# --- Configuration ---
@st.cache_resource
def load_config():
    return {
        "openai_api_key": st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
        "data_retention_days": 30,
        "similarity_threshold": 0.70,
        "model_name": "gpt-4",
        "embedding_model": "all-MiniLM-L6-v2"
    }

config = load_config()
openai.api_key = config["openai_api_key"]

# --- Database Setup with Error Handling ---
def init_database():
    try:
        conn = sqlite3.connect("chat_memory.db", check_same_thread=False)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_name TEXT,
            user_dept TEXT,
            question TEXT,
            answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        return conn
    except sqlite3.Error as e:
        st.error(f"Database error: {str(e)}")
        return None

conn = init_database()

# --- Model Loading with Caching ---
@st.cache_resource
def load_models():
    try:
        # Spell checking
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)
        
        # Sentence embeddings
        model = SentenceTransformer(config["embedding_model"])
        
        return sym_spell, model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

sym_spell, model = load_models()

# --- Data Loading with Validation ---
@st.cache_data
def load_qa_data():
    try:
        with open("qa_dataset.json", "r") as f:
            qa_data = json.load(f)
            questions = [item['question'] for item in qa_data]
            answers = [item['answer'] for item in qa_data]
            embeddings = model.encode(questions, convert_to_tensor=True) if model else None
            return questions, answers, embeddings
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Failed to load Q&A data: {str(e)}")
        return [], [], None

questions, answers, embeddings = load_qa_data()

# --- Text Processing Utilities ---
ABBREVIATIONS = {
    "u": "you", "ur": "your", "r": "are", "dept": "department",
    "info": "information", "sch": "school", "cn": "can"
}

def normalize_text(text):
    if not text or not isinstance(text, str):
        return ""
    words = text.split()
    return ' '.join([ABBREVIATIONS.get(w.lower(), w) for w in words])

def correct_spelling(text):
    try:
        if sym_spell and text:
            suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
            return suggestions[0].term if suggestions else text
        return text
    except Exception:
        return text

def preprocess_input(user_input):
    if not user_input:
        return ""
    text = normalize_text(user_input)
    return correct_spelling(text)

# --- Response Generation ---
def get_best_match(user_input):
    try:
        if not embeddings or not model:
            return 0, "", ""
            
        query_embedding = model.encode(user_input, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
        best_score = float(scores.max())
        best_idx = int(scores.argmax())
        return best_score, answers[best_idx], questions[best_idx]
    except Exception as e:
        st.error(f"Semantic search error: {str(e)}")
        return 0, "", ""

def generate_fallback_response(user_input):
    try:
        name = st.session_state.get("user_name", "a student")
        dept = st.session_state.get("user_dept", "the university")
        
        prompt = f"""You are a helpful assistant for Crescent University. 
        You are talking to {name} from {dept}. 
        Answer the following question professionally and concisely.
        
        Question: {user_input}
        Answer:"""
        
        response = openai.ChatCompletion.create(
            model=config["model_name"],
            messages=[{"role": "system", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        return response["choices"][0]["message"]["content"].strip()
    except openai.error.OpenAIError as e:
        st.error(f"API Error: {str(e)}")
        return "I'm having trouble connecting to the knowledge base. Please try again later."

# --- Database Operations ---
def store_conversation(session_id, question, answer):
    if not conn:
        return False
        
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO memory (session_id, user_name, user_dept, question, answer)
            VALUES (?, ?, ?, ?, ?)
        """, (
            session_id,
            st.session_state.get("user_name", ""),
            st.session_state.get("user_dept", ""),
            question,
            answer
        ))
        conn.commit()
        return True
    except sqlite3.Error:
        return False

def clean_old_records():
    if not conn:
        return
        
    try:
        c = conn.cursor()
        c.execute("""
            DELETE FROM memory 
            WHERE timestamp < datetime('now', ?)
        """, (f"-{config['data_retention_days']} days",))
        conn.commit()
    except sqlite3.Error:
        pass

# --- UI Components ---
def setup_page():
    st.set_page_config(
        page_title="🎓 Crescent Uni Assistant", 
        layout="wide",
        menu_items={
            'About': "Crescent University Chatbot v2.0"
        }
    )
    
    st.markdown("""
        <style>
        .message-bubble-user {
            background-color: #DCF8C6;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 5px;
            text-align: right;
        }
        .message-bubble-bot {
            background-color: #F1F0F0;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .bot-name {
            font-weight: bold;
            color: #4b4b4b;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

def user_profile_section():
    with st.expander("👤 Profile Settings", expanded=True):
        anonymous = st.checkbox("Chat anonymously", key="anonymous_mode")
        
        if not anonymous:
            st.session_state.user_name = st.text_input(
                "Your name:", 
                value=st.session_state.get("user_name", ""),
                key="user_name_input"
            )
            st.session_state.user_dept = st.text_input(
                "Your department:", 
                value=st.session_state.get("user_dept", ""),
                key="user_dept_input"
            )
        else:
            st.session_state.user_name = "Anonymous"
            st.session_state.user_dept = "Unknown"

def chat_interface():
    st.title("🎓 Crescent University Chatbot")
    st.markdown("👋 Welcome! Ask me anything about **admissions**, **courses**, **fees**, **departments**, or **campus life**.")
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Chat input
    user_query = st.text_area("💬 Your question:", key="user_input", height=100)
    
    col1, col2 = st.columns([4, 1])
    with col1:
        if st.button("Ask", key="ask_button"):
            process_user_query(user_query)
    with col2:
        if st.button("Clear History", key="clear_button"):
            st.session_state.history = []
    
    # Follow-up suggestions
    if st.session_state.history:
        with st.expander("🔍 Follow-up questions"):
            follow_up = st.radio(
                "Select a follow-up:",
                options=[
                    "",
                    "Can you explain more about this?",
                    "What are the requirements?",
                    "Are there related courses?"
                ],
                horizontal=True
            )
            if follow_up:
                process_user_query(f"{st.session_state.history[-1][0]} - {follow_up}")
    
    # Display history
    for q, a in reversed(st.session_state.history):
        st.markdown(f"<div class='message-bubble-user'>🙋‍♂️ {q}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='message-bubble-bot'><span class='bot-name'>🤖 Bot:</span> {a}</div>", unsafe_allow_html=True)

def process_user_query(user_query):
    if not user_query.strip():
        st.warning("Please enter a question")
        return
        
    with st.spinner("Searching for the best answer..."):
        # Preprocess input
        processed = preprocess_input(user_query)
        
        # Get best match from Q&A
        score, best_answer, matched_question = get_best_match(processed)
        
        # Generate response
        if score > config["similarity_threshold"]:
            answer = best_answer
        else:
            answer = generate_fallback_response(user_query)
        
        # Update session history
        st.session_state.history.append((user_query, answer))
        
        # Store in database
        store_conversation(str(time.time()), user_query, answer)

# --- Main Execution ---
if __name__ == "__main__":
    setup_page()
    clean_old_records()  # Run cleanup at startup
    user_profile_section()
    
    if not st.session_state.get("anonymous_mode", False) and not st.session_state.get("user_name", ""):
        st.info("Please set up your profile or enable anonymous mode to start chatting.")
    else:
        chat_interface()

    # Close database connection when done
    if conn:
        conn.close()
