# Enhanced Crescent University Chatbot
import streamlit as st
import json
import time
import sqlite3
import torch
import os
from sentence_transformers import SentenceTransformer, util
from symspellpy import SymSpell
import openai
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI
from langchain.memory import ConversationBufferMemory
import psycopg2
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="🎓 Crescent Uni Assistant Pro",
    layout="wide",
    menu_items={
        'About': "Crescent University Chatbot v4.0 (Enhanced)",
        'Get Help': 'mailto:support@crescent.edu',
        'Report a bug': "mailto:it-support@crescent.edu"
    }
)

# --- Config ---
class Config:
    def __init__(self):
        self.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        self.DATA_RETENTION_DAYS = 30
        self.SIMILARITY_THRESHOLD = 0.70
        self.LLM_MODEL = "gpt-3.5-turbo"
        self.EMBEDDING_MODEL = "all-MiniLM-L12-v2"
        self.SPELL_CHECKER_DICT = "frequency_dictionary_en_82_765.txt"
        self.MAX_CACHE_SIZE = 1000
        self.DB_BATCH_SIZE = 5
        self.IMAGE_DIR = "./university_images/"
        self.DOCS_DIR = "./university_docs/"
        self.ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin123")

config = Config()
openai.api_key = config.OPENAI_API_KEY

# --- Database Manager (PostgreSQL) ---
class DatabaseManager:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname=st.secrets.get("DB_NAME", "chatbot"),
            user=st.secrets.get("DB_USER", "postgres"),
            password=st.secrets.get("DB_PASSWORD", ""),
            host=st.secrets.get("DB_HOST", "localhost")
        )
        self._initialize_db()

    def _initialize_db(self):
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT,
                    user_name TEXT,
                    user_dept TEXT,
                    question TEXT,
                    answer TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    feedback INTEGER
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics (
                    id SERIAL PRIMARY KEY,
                    event_type TEXT,
                    event_data JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()

    def store_conversation(self, session_id, user_name, user_dept, question, answer):
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO memory (session_id, user_name, user_dept, question, answer)
                    VALUES (%s, %s, %s, %s, %s)
                """, (session_id, user_name, user_dept, question, answer))
                self.conn.commit()
            return True
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            return False

    def log_analytics(self, event_type, event_data):
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO analytics (event_type, event_data)
                    VALUES (%s, %s)
                """, (event_type, json.dumps(event_data)))
                self.conn.commit()
        except Exception as e:
            st.error(f"Analytics logging failed: {str(e)}")

    def __del__(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

# --- AI Service with RAG ---
class AIService:
    def __init__(self):
        self.embedding_model = load_embedding_model(config.EMBEDDING_MODEL)
        self.llm_model = config.LLM_MODEL
        self.qa_embeddings = None
        self.questions = []
        self.answers = []
        self.rag_index = None
        self._load_models()
        
    def _load_models(self):
        self.build_rag_index()
        self.load_qa_data()

    def build_rag_index(self):
        if os.path.exists(config.DOCS_DIR):
            documents = SimpleDirectoryReader(config.DOCS_DIR).load_data()
            llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
            service_context = ServiceContext.from_defaults(llm=llm)
            self.rag_index = VectorStoreIndex.from_documents(
                documents, 
                service_context=service_context
            )

    def query_rag(self, question):
        if self.rag_index:
            query_engine = self.rag_index.as_query_engine()
            return str(query_engine.query(question))
        return None

    def load_qa_data(self, qa_file="qa_data.json"):
        try:
            with open(qa_file, "r") as f:
                qa_data = json.load(f)
            
            self.questions = [item['question'] for item in qa_data]
            self.answers = [item['answer'] for item in qa_data]

            if self.questions:
                embeddings = self.embedding_model.encode(self.questions, convert_to_tensor=True)
                self.qa_embeddings = embeddings
            return True
        except Exception as e:
            st.error(f"Failed to load Q&A data: {str(e)}")
            return False

    def get_response(self, query, session_history=None):
        # Step 1: Check exact matches
        processed = self.preprocess_input(query)
        score, answer, matched = self.get_best_match(processed)
        
        if score > config.SIMILARITY_THRESHOLD:
            return answer, "qa_match"
            
        # Step 2: Try RAG system
        rag_response = self.query_rag(query)
        if rag_response and len(rag_response) > 20:  # Minimum length check
            return rag_response, "rag_system"
            
        # Step 3: Fallback to LLM with context
        context = self._build_context(session_history)
        prompt = f"{context}\n\nQuestion: {query}\nAnswer:"
        
        try:
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content, "llm_fallback"
        except Exception as e:
            return f"I'm sorry, I couldn't process your request. Error: {str(e)}", "error"

    def _build_context(self, session_history):
        base_context = """
        Crescent University Information:
        - Location: City, Country
        - Established: 2005
        - Key Departments: Computer Science, Engineering, Business
        - Current Academic Year: 2023-2024
        """
        
        if session_history:
            history_context = "\nRecent Conversation:\n" + "\n".join(
                f"Q: {q}\nA: {a}" for q, a in session_history[-3:]
            )
            return base_context + history_context
        return base_context

# --- Chat Interface with Memory ---
class ChatInterface:
    def __init__(self, ai_service, db_manager):
        self.ai = ai_service
        self.db = db_manager
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self._init_session_state()
        self._setup_ui_style()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _init_session_state(self):
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(time.time())
        if 'feedback' not in st.session_state:
            st.session_state.feedback = {}

    def _setup_ui_style(self):
        st.markdown("""
            <style>
                .message-bubble-user {
                    background-color: #f0f2f6;
                    padding: 10px 15px;
                    border-radius: 15px;
                    margin: 10px 0;
                    max-width: 80%;
                    float: right;
                    clear: both;
                }
                .message-bubble-bot {
                    background-color: #e3f2fd;
                    padding: 10px 15px;
                    border-radius: 15px;
                    margin: 10px 0;
                    max-width: 80%;
                    float: left;
                    clear: both;
                }
                .bot-name {
                    font-weight: bold;
                    color: #1976d2;
                }
                .stButton>button {
                    width: 100%;
                }
            </style>
        """, unsafe_allow_html=True)

    def show_chat_interface(self):
        st.title("🎓 Crescent University Assistant Pro")
        
        with st.expander("📌 Quick Questions", expanded=False):
            self._show_quick_questions()
            
        user_query = st.text_area("💬 Your question:", key="user_input", 
                                placeholder="Ask about admissions, courses, fees...")
        
        if st.button("Submit", key="ask_button"):
            self._process_user_query(user_query)
            
        self._display_chat_history()
        
        if st.session_state.history:
            self._show_feedback_options()

    def _process_user_query(self, query):
        if not query.strip():
            st.warning("Please enter a question")
            return
            
        # Add to history
        st.session_state.history.append((query, ""))
        
        # Process in background
        self.executor.submit(self._generate_and_display_response, query)
        
    def _generate_and_display_response(self, query):
        placeholder = st.empty()
        full_response = ""
        
        # Get response from AI service
        response, response_source = self.ai.get_response(query, st.session_state.history)
        
        # Stream the response for better UX
        for chunk in response.split():
            full_response += chunk + " "
            placeholder.markdown(
                f"<div class='message-bubble-bot'><span class='bot-name'>🤖 Assistant:</span> {full_response}</div>", 
                unsafe_allow_html=True
            )
            time.sleep(0.05)
            
        # Update history
        st.session_state.history[-1] = (query, full_response)
        
        # Store in database
        self.db.store_conversation(
            st.session_state.session_id,
            st.session_state.get("user_name", "Anonymous"),
            st.session_state.get("user_dept", "Unknown"),
            query,
            full_response
        )
        
        # Log analytics
        self.db.log_analytics("response_generated", {
            "query": query,
            "response_source": response_source,
            "response_length": len(full_response)
        })

    def _display_chat_history(self):
        for i, (q, a) in enumerate(st.session_state.history):
            st.markdown(f"<div class='message-bubble-user'>🙋‍♂️ {q}</div>", 
                       unsafe_allow_html=True)
            
            if a:  # Only show answer if it exists
                st.markdown(
                    f"<div class='message-bubble-bot'><span class='bot-name'>🤖 Assistant:</span> {a}</div>", 
                    unsafe_allow_html=True
                )
                
                # Show feedback buttons for each response
                if i not in st.session_state.feedback:
                    cols = st.columns(3)
                    with cols[0]:
                        if st.button("👍 Helpful", key=f"helpful_{i}"):
                            st.session_state.feedback[i] = 1
                    with cols[1]:
                        if st.button("👎 Not Helpful", key=f"not_helpful_{i}"):
                            st.session_state.feedback[i] = -1

    def _show_quick_questions(self):
        questions = [
            "What are the admission requirements?",
            "When is the application deadline?",
            "What programs are offered in Computer Science?",
            "How much is the tuition fee?",
            "What scholarships are available?"
        ]
        
        cols = st.columns(2)
        for i, q in enumerate(questions):
            with cols[i % 2]:
                if st.button(q, key=f"quick_q_{i}"):
                    self._process_user_query(q)

    def _show_feedback_options(self):
        st.markdown("---")
        with st.expander("📊 Conversation Feedback"):
            feedback = st.radio("How would you rate this conversation?", 
                              ["Excellent", "Good", "Average", "Poor"])
            if st.button("Submit Feedback"):
                self.db.log_analytics("conversation_feedback", {
                    "rating": feedback,
                    "session_id": st.session_state.session_id
                })
                st.success("Thank you for your feedback!")

# --- Admin Panel ---
class AdminPanel:
    def __init__(self, db_manager):
        self.db = db_manager
        
    def show(self):
        if not self._check_auth():
            return
            
        st.title("🔒 Admin Dashboard")
        tab1, tab2, tab3 = st.tabs(["Q&A Management", "Analytics", "System Settings"])
        
        with tab1:
            self._show_qa_management()
        with tab2:
            self._show_analytics()
        with tab3:
            self._show_system_settings()

    def _check_auth(self):
        if st.session_state.get("admin_authenticated"):
            return True
            
        password = st.text_input("Admin Password", type="password")
        if st.button("Login"):
            if password == config.ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        return False

    def _show_qa_management(self):
        st.header("Q&A Management")
        
        # Add new Q&A
        with st.form("new_qa_form"):
            new_q = st.text_input("New Question")
            new_a = st.text_area("Answer")
            if st.form_submit_button("Add Q&A Pair"):
                self._add_qa_pair(new_q, new_a)
                
        # View/Edit existing
        st.subheader("Existing Q&A Pairs")
        # Would implement actual database view/edit here

    def _show_analytics(self):
        st.header("Conversation Analytics")
        
        # Show basic stats
        with self.db.conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM memory")
            total_conversations = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(DISTINCT session_id) 
                FROM memory 
                WHERE timestamp > NOW() - INTERVAL '7 days'
            """)
            weekly_users = cursor.fetchone()[0]
            
        st.metric("Total Conversations", total_conversations)
        st.metric("Weekly Active Users", weekly_users)
        
        # More analytics would be added here

# --- Main Application ---
def main():
    # Initialize services
    db_manager = DatabaseManager()
    ai_service = AIService()
    
    # Check if admin view requested
    if st.experimental_get_query_params().get("admin"):
        AdminPanel(db_manager).show()
        return
        
    # Main chat interface
    chat_interface = ChatInterface(ai_service, db_manager)
    
    # Layout
    with st.sidebar:
        st.image("https://via.placeholder.com/200x50?text=Crescent+University", width=200)
        chat_interface._show_profile_section()
        st.markdown("---")
        st.markdown("**Need help?** [Contact support](mailto:support@crescent.edu)")
        
        if st.button("Admin Login"):
            st.experimental_set_query_params(admin=True)
            st.rerun()
    
    # Main chat area
    chat_interface.show_chat_interface()

if __name__ == "__main__":
    main()
