# --- Imports ---
import streamlit as st
import json
import time
import os
import openai
import sqlite3
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from textblob import TextBlob
from symspellpy.symspellpy import SymSpell
import pkg_resources

# --- Page Config ---
st.set_page_config(page_title="🎓 Crescent Uni Assistant Pro", layout="wide")

# --- Config ---
class Config:
    def __init__(self):
        self.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        self.SIMILARITY_THRESHOLD = 0.65
        self.LLM_MODEL = "gpt-3.5-turbo"
        self.EMBEDDING_MODEL = "all-MiniLM-L12-v2"
        self.DOCS_DIR = "./university_docs/"
        self.ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin123")
        self.SYMSPELL_DICT_PATH = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")

config = Config()
openai.api_key = config.OPENAI_API_KEY

# --- Database Manager (SQLite) ---
class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect("chatbot.db", check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            self.conn.execute("""
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
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    event_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def store_conversation(self, session_id, user_name, user_dept, question, answer):
        with self.conn:
            self.conn.execute("""
                INSERT INTO memory (session_id, user_name, user_dept, question, answer)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, user_name, user_dept, question, answer))

    def log_analytics(self, event_type, event_data):
        with self.conn:
            self.conn.execute("""
                INSERT INTO analytics (event_type, event_data)
                VALUES (?, ?)
            """, (event_type, json.dumps(event_data)))

# --- AI Service ---
def load_embedding_model(name):
    return SentenceTransformer(name)

def detect_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity < -0.4:
        return "negative"
    elif polarity > 0.4:
        return "positive"
    return "neutral"

ABBREVIATIONS = {
    "u": "you", "r": "are", "ur": "your", "cn": "can", "pls": "please", "asap": "as soon as possible"
}

SYNONYMS = {
    "school fees": "tuition",
    "hostel": "accommodation",
    "it office": "ict department",
    "lecturers": "academic staff",
    "non-academic staff": "administrative staff",
    "courses": "programs",
    "bio": "biochemistry",
    "comp sci": "computer science"
}

class AIService:
    def __init__(self):
        self.embedding_model = load_embedding_model(config.EMBEDDING_MODEL)
        self.llm_model = config.LLM_MODEL
        self.questions = []
        self.answers = []
        self.qa_embeddings = None
        self.rag_index = None
        self.symspell = self.initialize_symspell()
        self.load_qa_data()
        self.build_rag_index()

    def initialize_symspell(self):
        symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        symspell.load_dictionary(config.SYMSPELL_DICT_PATH, term_index=0, count_index=1)
        return symspell

    def correct_spelling(self, text):
        corrected = []
        for word in text.split():
            suggestions = self.symspell.lookup(word, verbosity=2)
            corrected.append(suggestions[0].term if suggestions else word)
        return " ".join(corrected)

    def expand_abbreviations(self, text):
        words = text.split()
        expanded = [ABBREVIATIONS.get(word, word) for word in words]
        return " ".join(expanded)

    def replace_synonyms(self, text):
        for key, val in SYNONYMS.items():
            text = text.replace(key, val)
        return text

    def preprocess_input(self, query):
        query = query.lower().strip()
        query = self.correct_spelling(query)
        query = self.expand_abbreviations(query)
        query = self.replace_synonyms(query)
        return query

    def load_qa_data(self):
        try:
            with open("qa_data.json", "r") as f:
                data = json.load(f)
                self.questions = [self.preprocess_input(item['question']) for item in data]
                self.answers = [item['answer'] for item in data]
                self.qa_embeddings = self.embedding_model.encode(self.questions, convert_to_tensor=True)
        except Exception as e:
            st.error(f"Failed to load QA data: {str(e)}")

    def build_rag_index(self):
        if os.path.exists(config.DOCS_DIR):
            documents = SimpleDirectoryReader(config.DOCS_DIR).load_data()
            llm = LlamaOpenAI(model=config.LLM_MODEL, temperature=0.2)
            service_context = ServiceContext.from_defaults(llm=llm)
            self.rag_index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    def is_followup(self, query):
        return any(word in query.lower() for word in ["what about", "how about", "that one", "those", "and fees", "the requirement"])

    def get_best_match(self, query):
        if self.qa_embeddings is None:
            return 0, None, None
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.qa_embeddings)[0]
        best_score = float(scores.max())
        best_index = int(scores.argmax())
        return best_score, self.answers[best_index], self.questions[best_index]

    def query_rag(self, query):
        if self.rag_index:
            engine = self.rag_index.as_query_engine()
            return str(engine.query(query))
        return None

    def detect_small_talk(self, query):
        small_talk = {
            "hi": "Hello there! 😊",
            "hello": "Hi! How can I assist you today?",
            "how are you": "I’m doing great! Excited to help you.",
            "thank you": "You're very welcome! 🙏",
            "thanks": "Anytime 😊"
        }
        for trigger, reply in small_talk.items():
            if trigger in query.lower():
                return reply
        return None

    def get_response(self, query, session_history=None):
        query = self.preprocess_input(query)

        small_talk = self.detect_small_talk(query)
        if small_talk:
            return small_talk, "small_talk"

        if self.is_followup(query) and session_history:
            last_topic = session_history[-1][0]
            query = f"{last_topic}. Follow-up: {query}"

        score, answer, match = self.get_best_match(query)
        if score > config.SIMILARITY_THRESHOLD:
            return answer, "qa_match"

        rag_answer = self.query_rag(query)
        if rag_answer and len(rag_answer) > 20:
            return rag_answer, "rag_system"

        if score < config.SIMILARITY_THRESHOLD and (not rag_answer or len(rag_answer) < 20):
            return "I'm not sure I fully understood that 🤔. Could you clarify or ask in another way?", "clarification"

        context = "\n".join([f"Q: {q}\nA: {a}" for q, a in (session_history[-3:] if session_history else [])])
        sentiment = detect_sentiment(query)
        if sentiment == "negative":
            tone_prefix = "I'm really sorry you're having trouble. Let’s see how I can help 💙.\n"
        elif sentiment == "positive":
            tone_prefix = "Great to hear from you! 😄\n"
        else:
            tone_prefix = ""

        prompt = f"{tone_prefix}You are a helpful, friendly, and slightly humorous assistant for Crescent University.\nAlways keep responses clear, short, and encouraging.\n\n{context}\n\nUser: {query}\nAssistant:"

        try:
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=600,
                stream=False
            )
            return response["choices"][0]["message"]["content"], "llm"
        except Exception as e:
            return f"[LLM Error: {str(e)}]", "error"

# --- Chat Interface ---
class ChatInterface:
    def __init__(self, ai_service, db):
        self.ai = ai_service
        self.db = db
        self.init_session()

    def init_session(self):
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(time.time())
        if 'user_name' not in st.session_state:
            st.session_state.user_name = st.text_input("Enter your name", "Anonymous")
        if 'user_dept' not in st.session_state:
            st.session_state.user_dept = st.selectbox("Your department", ["General", "Computer Science", "Mass Comm", "Biochem", "Other"])

    def show(self):
        st.title(":sparkles: Crescent University Assistant Pro")

        prompt = st.chat_input("Ask me anything about the university...")
        if prompt:
            st.session_state.history.append((prompt, ""))
            self.respond(prompt)

        for q, a in st.session_state.history:
            with st.chat_message("user"):
                st.markdown(q)
            if a:
                with st.chat_message("assistant"):
                    st.markdown(a)

    def respond(self, query):
        full_response = ""
        response, source = self.ai.get_response(query, st.session_state.history)

        full_response = response
        with st.chat_message("assistant"):
            st.markdown(full_response)

        st.session_state.history[-1] = (query, full_response)
        self.db.store_conversation(
            st.session_state.session_id,
            st.session_state.user_name,
            st.session_state.user_dept,
            query,
            full_response
        )
        self.db.log_analytics("response", {
            "query": query, "source": source, "length": len(full_response)
        })

# --- Main App ---
def main():
    db = DatabaseManager()
    ai = AIService()
    chat = ChatInterface(ai, db)
    chat.show()

if __name__ == "__main__":
    main()
