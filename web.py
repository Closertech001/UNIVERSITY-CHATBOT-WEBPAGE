# Cleaned version with fixes applied
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

# --- Page Configuration ---
st.set_page_config(
    page_title="🎓 Crescent Uni Assistant",
    layout="wide",
    menu_items={
        'About': "Crescent University Chatbot v3.0 (Optimized)",
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
        self.LLM_MODEL = "gpt-4"
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.SPELL_CHECKER_DICT = "frequency_dictionary_en_82_765.txt"
        self.MAX_CACHE_SIZE = 1000
        self.DB_BATCH_SIZE = 5

config = Config()
openai.api_key = config.OPENAI_API_KEY

# --- Embedding Model Loader ---
@st.cache_resource
def load_embedding_model(name):
    return SentenceTransformer(name, device='cpu', cache_folder='./model_cache')

# --- Q&A Data Loader ---
@st.cache_data
def load_qa_file(qa_file):
    with open(qa_file, "r") as f:
        return json.load(f)

# --- Text Processor ---
class TextProcessor:
    ABBREVIATIONS = {
        "u": "you", "ur": "your", "r": "are", "dept": "department",
        "info": "information", "sch": "school", "cn": "can"
    }

    def __init__(self, spell_checker):
        self.spell_checker = spell_checker

    def normalize_text(self, text):
        if not text or not isinstance(text, str):
            return ""
        words = text.split()
        return ' '.join([self.ABBREVIATIONS.get(w.lower(), w) for w in words])

    def correct_spelling(self, text):
        try:
            if self.spell_checker and text:
                suggestions = self.spell_checker.lookup_compound(text, max_edit_distance=2)
                return suggestions[0].term if suggestions else text
            return text
        except Exception:
            return text

    def preprocess_input(self, user_input):
        if not user_input:
            return ""
        text = self.normalize_text(user_input)
        return self.correct_spelling(text)

# --- AI Service ---
class AIService:
    def __init__(self, embedding_model_name, llm_model):
        self.embedding_model = load_embedding_model(embedding_model_name)
        self.llm_model = llm_model
        self.qa_embeddings = None
        self.questions = []
        self.answers = []
        self.common_responses = {
            "hello": "Hello! How can I help you today?",
            "hi": "Hi there! What would you like to know about Crescent University?",
            "thanks": "You're welcome! Is there anything else I can help with?",
            "thank you": "You're welcome! Is there anything else I can help with?",
            "goodbye": "Goodbye! Have a great day!",
            "bye": "Goodbye! Come back if you have more questions!"
        }

    def load_qa_data(self, qa_file="qa_data.json"):
        try:
            qa_data = load_qa_file(qa_file)
            self.questions = [item['question'] for item in qa_data]
            self.answers = [item['answer'] for item in qa_data]

            if not self.questions:
                st.warning("Q&A data is empty")
                return False

            batch_size = 32
            embeddings = [
                self.embedding_model.encode(self.questions[i:i+batch_size], convert_to_tensor=True)
                for i in range(0, len(self.questions), batch_size)
            ]
            self.qa_embeddings = torch.cat(embeddings) if len(embeddings) > 1 else embeddings[0]
            return True
        except Exception as e:
            st.error(f"Failed to load Q&A data: {e}")
            return False

    @lru_cache(maxsize=config.MAX_CACHE_SIZE)
    def get_cached_response(self, processed_query):
        query_embedding = self.embedding_model.encode(processed_query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, self.qa_embeddings)[0]
        best_score = float(scores.max())
        best_idx = int(scores.argmax())
        return best_score, self.answers[best_idx], self.questions[best_idx]

    def get_best_match(self, user_input):
        if self.qa_embeddings is None:
            return 0, "", ""

        lower_input = user_input.lower().strip()
        if lower_input in self.common_responses:
            return 1.0, self.common_responses[lower_input], lower_input

        return self.get_cached_response(user_input)

    def generate_fallback_response(self, user_input, user_name, user_dept):
        if not config.OPENAI_API_KEY:
            return "I'm temporarily unavailable due to missing API key."
        try:
            prompt = f"As Crescent University's assistant, provide accurate information to {user_name} ({user_dept}). Question: {user_input} Answer concisely:"
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"AI error: {str(e)}"

# --- Chat Interface ---
class ChatInterface:
    def __init__(self):
        self._init_session_state()
        self._setup_ui_style()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _init_session_state(self):
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'user_name' not in st.session_state:
            st.session_state.user_name = ""
        if 'user_dept' not in st.session_state:
            st.session_state.user_dept = ""
        if 'anonymous_mode' not in st.session_state:
            st.session_state.anonymous_mode = False

    def _setup_ui_style(self):
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
            .stTextArea textarea {
                min-height: 100px;
            }
            .stSpinner > div {
                text-align: center;
            }
            </style>
        """, unsafe_allow_html=True)

    def show_profile_section(self):
        with st.expander("👤 Profile Settings", expanded=True):
            st.session_state.anonymous_mode = st.checkbox(
                "Chat anonymously", 
                value=st.session_state.anonymous_mode
            )
            
            if not st.session_state.anonymous_mode:
                st.session_state.user_name = st.text_input(
                    "Your name:", 
                    value=st.session_state.user_name
                )
                st.session_state.user_dept = st.text_input(
                    "Your department:", 
                    value=st.session_state.user_dept
                )
            else:
                st.session_state.user_name = "Anonymous"
                st.session_state.user_dept = "Unknown"

    def show_chat_interface(self, ai_service, text_processor, db_manager):
        st.title("🎓 Crescent University Chatbot")
        st.markdown("👋 Welcome! Ask me anything about **admissions**, **courses**, **fees**, **departments**, or **campus life**.")

        user_query = st.text_area("💬 Your question:", key="user_input", placeholder="Type your question here...")

        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button("Ask", key="ask_button"):
                self._process_user_query(user_query, ai_service, text_processor, db_manager)
        with col2:
            if st.button("Clear History", key="clear_button"):
                st.session_state.history = []
                st.experimental_rerun()

        self._show_follow_up_suggestions(ai_service, text_processor, db_manager)
        self._display_chat_history()

    def _process_user_query(self, user_query, ai_service, text_processor, db_manager):
        if not user_query.strip():
            st.warning("Please enter a question")
            return
            
        # Add message immediately
        st.session_state.history.append((user_query, "Thinking..."))
        self._display_chat_history()
        
        # Process in background
        def process_query():
            processed = text_processor.preprocess_input(user_query)
            score, best_answer, matched_question = ai_service.get_best_match(processed)
            
            if score > config.SIMILARITY_THRESHOLD:
                return best_answer
            return ai_service.generate_fallback_response(
                user_query,
                st.session_state.user_name,
                st.session_state.user_dept
            )
        
        # Run in thread and update when done
        def update_answer():
            answer = process_query()
            st.session_state.history[-1] = (user_query, answer)
            if db_manager:
                db_manager.store_conversation(
                    str(time.time()),
                    st.session_state.user_name,
                    st.session_state.user_dept,
                    user_query,
                    answer
                )
            st.experimental_rerun()
            
        self.executor.submit(update_answer)

    def _show_follow_up_suggestions(self, ai_service, text_processor, db_manager):
        if st.session_state.history:
            with st.expander("🔍 Follow-up questions"):
                follow_up = st.radio(
                    "Select a follow-up:",
                    options=[
                        "",
                        "Can you explain more about this?",
                        "What are the requirements?",
                        "Are there related courses?",
                        "When is the deadline?"
                    ],
                    horizontal=True
                )
                if follow_up:
                    self._process_user_query(
                        f"{st.session_state.history[-1][0]} - {follow_up}",
                        ai_service,
                        text_processor,
                        db_manager
                    )

    def _display_chat_history(self):
        for q, a in reversed(st.session_state.history):
            st.markdown(f"<div class='message-bubble-user'>🙋‍♂️ {q}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='message-bubble-bot'><span class='bot-name'>🤖 Bot:</span> {a}</div>", unsafe_allow_html=True)

# --- Main Application ---
def main():
    # Initialize spell checker
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym_spell.load_dictionary(config.SPELL_CHECKER_DICT, 0, 1)
    
    # Initialize services
    text_processor = TextProcessor(sym_spell)
    ai_service = AIService(config.EMBEDDING_MODEL, config.LLM_MODEL)
    
    # Load Q&A data
    if not ai_service.load_qa_data():
        st.error("Failed to initialize knowledge base. Some functionality may be limited.")
        st.stop()
    
    # Initialize database and clean old records
    with DatabaseManager() as db_manager:
        if db_manager:
            db_manager.clean_old_records(config.DATA_RETENTION_DAYS)
        
        # Initialize chat interface
        chat_interface = ChatInterface()
        
        # Split layout into sidebar and main area
        with st.sidebar:
            st.image("https://via.placeholder.com/200x50?text=Crescent+University", width=200)
            chat_interface.show_profile_section()
            st.markdown("---")
            st.markdown("**About this assistant:**")
            st.markdown("""
                - Answers questions about Crescent University
                - Provides admission and course information
                - Maintains conversation history
            """)
        
        # Main chat interface
        chat_interface.show_chat_interface(ai_service, text_processor, db_manager)

if __name__ == "__main__":
    main()
