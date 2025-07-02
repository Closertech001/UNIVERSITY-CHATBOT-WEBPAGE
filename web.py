# --- Imports ---
import streamlit as st
import json
import time
import os
import openai
import psycopg2
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI as LlamaOpenAI
from textblob import TextBlob
from symspellpy.symspellpy import SymSpell, Verbosity

# --- Page Config ---
st.set_page_config(page_title="🎓 Crescent Uni Assistant Pro", layout="wide")

# --- Config ---
class Config:
    def __init__(self):
        self.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        self.SIMILARITY_THRESHOLD = 0.70
        self.LOW_CONFIDENCE_THRESHOLD = 0.4
        self.LLM_MODEL = "gpt-3.5-turbo"
        self.EMBEDDING_MODEL = "all-MiniLM-L12-v2"
        self.DOCS_DIR = "./university_docs/"
        self.ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin123")

config = Config()
openai.api_key = config.OPENAI_API_KEY

# --- Database Manager ---
class DatabaseManager:
    def __init__(self):
        try:
            self.conn = psycopg2.connect(
                dbname=st.secrets.get("DB_NAME", "chatbot"),
                user=st.secrets.get("DB_USER", "postgres"),
                password=st.secrets.get("DB_PASSWORD", ""),
                host=st.secrets.get("DB_HOST", "localhost")
            )
            self.create_tables()
        except Exception as e:
            st.warning(f"Database connection failed: {str(e)}. Continuing without DB.")
            self.conn = None

    def create_tables(self):
        if not self.conn:
            return
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT,
                    user_name TEXT,
                    user_dept TEXT,
                    question TEXT,
                    answer TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics (
                    id SERIAL PRIMARY KEY,
                    event_type TEXT,
                    event_data JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self.conn.commit()

    def store_conversation(self, session_id, user_name, user_dept, question, answer):
        if not self.conn:
            return
        with self.conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO memory (session_id, user_name, user_dept, question, answer)
                VALUES (%s, %s, %s, %s, %s)
            """, (session_id, user_name, user_dept, question, answer))
            self.conn.commit()

    def log_analytics(self, event_type, event_data):
        if not self.conn:
            return
        with self.conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO analytics (event_type, event_data)
                VALUES (%s, %s)
            """, (event_type, json.dumps(event_data)))
            self.conn.commit()

# --- AI Service ---
class AIService:
    def __init__(self):
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.llm_model = config.LLM_MODEL
        self.questions = []
        self.answers = []
        self.qa_embeddings = None
        self.rag_index = None
        self.load_qa_data()
        self.build_rag_index()
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        # Load dictionary file for SymSpell (make sure the path/file is correct)
        dict_path = "frequency_dictionary_en_82_765.txt"
        if os.path.exists(dict_path):
            self.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
        else:
            st.warning(f"SymSpell dictionary file '{dict_path}' not found. Spell correction disabled.")

        # Abbreviations and synonyms map
        self.ABBREVIATIONS = {
            "u": "you",
            "r": "are",
            "ur": "your",
            "dept": "department",
            "uni": "university",
            "admis": "admission",
            "fac": "faculty",
            "lect": "lecturer",
            "prof": "professor",
            "asap": "as soon as possible",
            # Add more as needed
        }

    def load_qa_data(self):
        try:
            with open("qa_data.json", "r") as f:
                data = json.load(f)
                self.questions = [item['question'] for item in data]
                self.answers = [item['answer'] for item in data]
                self.qa_embeddings = self.embedding_model.encode(self.questions, convert_to_tensor=True)
        except Exception as e:
            st.error(f"Failed to load QA data: {str(e)}")
            self.questions = []
            self.answers = []
            self.qa_embeddings = None

    def build_rag_index(self):
        if os.path.exists(config.DOCS_DIR):
            documents = SimpleDirectoryReader(config.DOCS_DIR).load_data()
            llm = LlamaOpenAI(model=config.LLM_MODEL, temperature=0.2)
            service_context = ServiceContext.from_defaults(llm=llm)
            self.rag_index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        else:
            self.rag_index = None

    def preprocess_input(self, query):
        # Spell correction via SymSpell
        if hasattr(self, "sym_spell") and self.sym_spell:
            suggestions = self.sym_spell.lookup_compound(query, max_edit_distance=2)
            if suggestions:
                query = suggestions[0].term

        # Replace abbreviations
        words = query.split()
        words = [self.ABBREVIATIONS.get(w.lower(), w) for w in words]
        return " ".join(words).lower().strip()

    def get_best_match(self, query):
        if not self.qa_embeddings or not self.questions:
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
        small_talk_triggers = ["hi", "hello", "hey", "good morning", "good afternoon", "how are you"]
        for phrase in small_talk_triggers:
            if phrase in query.lower():
                user_name = st.session_state.get("user_name", "there")
                return f"Hello {user_name}! 😊 How can I assist you today?"
        return None

    def get_response(self, query, session_history=None):
        query = self.preprocess_input(query)

        # Detect small talk first
        small_talk = self.detect_small_talk(query)
        if small_talk:
            return small_talk, "small_talk"

        # Get best similarity match
        score, answer, match = self.get_best_match(query)
        if score > config.SIMILARITY_THRESHOLD:
            return answer, "qa_match"

        # Try RAG retrieval
        rag_answer = self.query_rag(query)
        if rag_answer and len(rag_answer) > 20:
            return rag_answer, "rag_system"

        # Low confidence: ask for clarification
        if score < config.LOW_CONFIDENCE_THRESHOLD:
            return "I’m not quite sure I understand. Could you please clarify or rephrase your question?", "clarification"

        # Sentiment analysis for tone
        sentiment = TextBlob(query).sentiment.polarity
        tone = ""
        if sentiment < -0.3:
            tone = "I'm sorry to hear that. I'll do my best to help you."
        elif sentiment > 0.3:
            tone = "That's great! How can I assist you further?"

        # Prepare multi-turn context
        context = ""
        if session_history:
            last_pairs = session_history[-3:]
            context = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in last_pairs if a])

        prompt = f"You are a helpful, friendly, and slightly humorous assistant for Crescent University.\n" \
                 f"{tone}\nConversation history:\n{context}\n\nUser: {query}\nAssistant:"

        try:
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=600,
                stream=True
            )
            return response, "llm_stream"
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
            st.session_state.user_name = ""

    def show(self):
        st.title(":sparkles: Crescent University Assistant Pro")

        # Ask for user name if not set
        if not st.session_state.user_name:
            name = st.text_input("Hi! What's your name?", key="user_name_input")
            if name:
                st.session_state.user_name = name.strip()
                st.success(f"Nice to meet you, {st.session_state.user_name}!")
            st.stop()

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

        if source == "llm_stream":
            try:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    for chunk in response:
                        token = chunk.choices[0].delta.get("content", "")
                        full_response += token
                        message_placeholder.markdown(full_response + "|")
                        time.sleep(0.02)
            except Exception as e:
                full_response = f"[Stream Error: {str(e)}]"
                st.error(full_response)
        else:
            full_response = response
            with st.chat_message("assistant"):
                st.markdown(full_response)

        # Update last message in history with response
        st.session_state.history[-1] = (query, full_response)

        # Store conversation & log analytics if DB is connected
        self.db.store_conversation(
            st.session_state.session_id,
            st.session_state.user_name,
            "General",
            query,
            full_response
        )
        self.db.log_analytics("response", {
            "query": query,
            "source": source,
            "length": len(full_response)
        })

# --- Main App ---
def main():
    db = DatabaseManager()
    ai = AIService()
    chat = ChatInterface(ai, db)
    chat.show()

if __name__ == "__main__":
    main()
