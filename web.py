# web.py - Crescent University Chatbot (Streamlit + Supabase + GPT + RAG)
import streamlit as st
import openai
import os
import json
import time
import uuid
import random
import psycopg2
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob
from symspellpy.symspellpy import SymSpell, Verbosity
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from dotenv import load_dotenv

# --- Load secrets and environment ---
env_path = ".env"
if os.path.exists(env_path):
    load_dotenv(env_path)

DB_NAME = st.secrets.get("DB_NAME") or os.getenv("DB_NAME")
DB_USER = st.secrets.get("DB_USER") or os.getenv("DB_USER")
DB_PASSWORD = st.secrets.get("DB_PASSWORD") or os.getenv("DB_PASSWORD")
DB_HOST = st.secrets.get("DB_HOST") or os.getenv("DB_HOST")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD") or os.getenv("ADMIN_PASSWORD", "admin123")

openai.api_key = OPENAI_API_KEY

# --- Config ---
class Config:
    LLM_MODEL = "gpt-3.5-turbo"
    EMBEDDING_MODEL = "all-MiniLM-L12-v2"
    SIMILARITY_THRESHOLD = 0.7
    LOW_CONFIDENCE_THRESHOLD = 0.4
    DOCS_DIR = "./university_docs/"

config = Config()

# --- DB Manager ---
class DatabaseManager:
    def __init__(self):
        try:
            self.conn = psycopg2.connect(
                dbname=DB_NAME, user=DB_USER,
                password=DB_PASSWORD, host=DB_HOST
            )
            self.create_tables()
        except Exception as e:
            st.warning(f"⚠️ DB connection failed: {e}")
            self.conn = None

    def create_tables(self):
        if not self.conn: return
        with self.conn.cursor() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT, user_name TEXT, user_dept TEXT,
                    question TEXT, answer TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS analytics (
                    id SERIAL PRIMARY KEY,
                    event_type TEXT, event_data JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self.conn.commit()

    def store_conversation(self, session_id, user_name, user_dept, question, answer):
        if self.conn:
            with self.conn.cursor() as c:
                c.execute("""INSERT INTO memory (session_id, user_name, user_dept, question, answer)
                             VALUES (%s, %s, %s, %s, %s)""",
                          (session_id, user_name, user_dept, question, answer))
                self.conn.commit()

    def log_analytics(self, event_type, event_data):
        if self.conn:
            with self.conn.cursor() as c:
                c.execute("""INSERT INTO analytics (event_type, event_data)
                             VALUES (%s, %s)""", (event_type, json.dumps(event_data)))
                self.conn.commit()

# --- AI Service ---
class AIService:
    def __init__(self, db=None):
        self.db = db
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.llm_model = config.LLM_MODEL
        self.questions, self.answers = [], []
        self.qa_embeddings = None
        self.rag_index = None
        self.load_qa_data()
        self.build_rag_index()
        self.sym_spell = self.init_symspell()

        self.ABBREVIATIONS = {
            "u": "you", "r": "are", "ur": "your", "dept": "department",
            "uni": "university", "admis": "admission", "fac": "faculty",
            "lect": "lecturer", "prof": "professor", "asap": "as soon as possible"
        }

    def init_symspell(self):
        sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dict_path = "frequency_dictionary_en_82_765.txt"
        if os.path.exists(dict_path):
            sym.load_dictionary(dict_path, term_index=0, count_index=1)
            return sym
        else:
            st.warning("⚠️ SymSpell dictionary not found. Spell correction disabled.")
            return None

    def load_qa_data(self):
        try:
            with open("qa_data.json", "r") as f:
                data = json.load(f)
                self.questions = [d['question'] for d in data]
                self.answers = [d['answer'] for d in data]
                self.qa_embeddings = self.embedding_model.encode(self.questions, convert_to_tensor=True)
        except Exception as e:
            st.error(f"❌ Failed to load QA data: {e}")

    def build_rag_index(self):
        try:
            if os.path.exists(config.DOCS_DIR) and os.listdir(config.DOCS_DIR):
                docs = SimpleDirectoryReader(config.DOCS_DIR).load_data()
                llm = LlamaOpenAI(model=config.LLM_MODEL, temperature=0.2)
                ctx = ServiceContext.from_defaults(llm=llm)
                self.rag_index = VectorStoreIndex.from_documents(docs, service_context=ctx)
        except Exception as e:
            st.warning(f"⚠️ RAG build failed: {e}")

    def preprocess_input(self, query):
        original = query
        corrected = None
        if self.sym_spell:
            suggestions = self.sym_spell.lookup_compound(query, max_edit_distance=2)
            if suggestions:
                corrected = suggestions[0].term
                query = corrected

        words = query.split()
        query = " ".join([self.ABBREVIATIONS.get(w.lower(), w) for w in words])
        if self.db:
            self.db.log_analytics("typo_abbreviation", {
                "original": original, "corrected": corrected, "final": query
            })
        return query

    def get_best_match(self, query):
        if not self.qa_embeddings: return 0, None, None
        q_embed = self.embedding_model.encode(query, convert_to_tensor=True)
        sims = util.cos_sim(q_embed, self.qa_embeddings)[0]
        score = float(sims.max())
        index = int(sims.argmax())
        return score, self.answers[index], self.questions[index]

    def query_rag(self, query):
        if self.rag_index:
            try:
                engine = self.rag_index.as_query_engine()
                return str(engine.query(query))
            except:
                return None
        return None

    def detect_small_talk(self, q):
        greetings = {
            "hi": "Hello!", "hello": "Hi!", "hey": "Hey!", "bye": "Goodbye!",
            "thanks": "You're welcome!", "thank you": "Glad I could help!"
        }
        for k, v in greetings.items():
            if k in q.lower():
                return v
        return None

    def get_response(self, query, history=None):
        query = self.preprocess_input(query)

        if (reply := self.detect_small_talk(query)):
            return reply, "small_talk"

        score, answer, _ = self.get_best_match(query)
        if score > config.SIMILARITY_THRESHOLD:
            return answer, "qa_match"

        rag_result = self.query_rag(query)
        if rag_result and len(rag_result) > 20:
            return rag_result, "rag"

        context = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in (history or [])[-3:]])
        prompt = [
            {"role": "system", "content": "You are a friendly, helpful assistant at Crescent University."},
            {"role": "user", "content": f"Context:\n{context}\n\nUser: {query}"}
        ]

        try:
            stream = openai.ChatCompletion.create(
                model=self.llm_model, messages=prompt,
                temperature=0.4, max_tokens=600, stream=True
            )
            return stream, "llm_stream"
        except:
            return "Sorry, something went wrong.", "error"

# --- Chat UI ---
class ChatInterface:
    def __init__(self, ai: AIService, db: DatabaseManager):
        self.ai = ai
        self.db = db
        self.init_session()

    def init_session(self):
        for k, v in {
            "history": [], "session_id": str(uuid.uuid4()),
            "user_name": "", "user_dept": "General"
        }.items():
            if k not in st.session_state:
                st.session_state[k] = v

    def show(self):
        st.title("🎓 Crescent University Chatbot")

        if not st.session_state.user_name:
            name = st.text_input("Hi! What's your name?")
            if name:
                st.session_state.user_name = name
                st.success(f"Welcome, {name}!")
            else:
                st.stop()

        st.session_state.user_dept = st.selectbox(
            "Select your department:",
            ["General", "Computer Science", "Business", "Engineering", "Law"]
        )

        if st.button("🔁 Reset"):
            st.session_state.history = []
            st.experimental_rerun()

        prompt = st.chat_input("Ask anything...")
        if prompt:
            st.session_state.history.append((prompt, ""))
            self.respond(prompt)

        for q, a in st.session_state.history:
            with st.chat_message("user"): st.markdown(q)
            if a:
                with st.chat_message("assistant"): st.markdown(a)

    def respond(self, query):
        response, source = self.ai.get_response(query, st.session_state.history)
        full = ""

        if source == "llm_stream":
            with st.chat_message("assistant"):
                placeholder = st.empty()
                for chunk in response:
                    token = chunk.choices[0].delta.get("content", "")
                    full += token
                    placeholder.markdown(full + "▌")
                    time.sleep(0.02)
                placeholder.markdown(full)
        else:
            full = response
            with st.chat_message("assistant"): st.markdown(full)

        st.session_state.history[-1] = (query, full)
        self.db.store_conversation(
            st.session_state.session_id,
            st.session_state.user_name,
            st.session_state.user_dept,
            query,
            full
        )
        self.db.log_analytics("response", {
            "query": query, "source": source,
            "length": len(full), "sentiment": TextBlob(query).sentiment.polarity
        })

# --- Main ---
def main():
    db = DatabaseManager()
    ai = AIService(db=db)
    chat = ChatInterface(ai, db)
    chat.show()

if __name__ == "__main__":
    main()
