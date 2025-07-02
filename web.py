# --- Imports ---
import streamlit as st
import json
import time
import os
import openai
import psycopg2
import random
import uuid
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
        self.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.SIMILARITY_THRESHOLD = 0.70
        self.LOW_CONFIDENCE_THRESHOLD = 0.4
        self.LLM_MODEL = "gpt-3.5-turbo"
        self.EMBEDDING_MODEL = "all-MiniLM-L12-v2"
        self.DOCS_DIR = "./university_docs/"
        self.ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD") or "admin123"

config = Config()
openai.api_key = config.OPENAI_API_KEY

# --- Database Manager ---
class DatabaseManager:
    def __init__(self):
        try:
            self.conn = psycopg2.connect(
                dbname=st.secrets.get("DB_NAME") or "chatbot",
                user=st.secrets.get("DB_USER") or "postgres",
                password=st.secrets.get("DB_PASSWORD") or "",
                host=st.secrets.get("DB_HOST") or "localhost"
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
    def __init__(self, db=None):
        self.db = db
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.llm_model = config.LLM_MODEL
        self.questions = []
        self.answers = []
        self.qa_embeddings = None
        self.rag_index = None
        self.load_qa_data()
        self.build_rag_index()

        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dict_path = "frequency_dictionary_en_82_765.txt"
        if os.path.exists(dict_path):
            self.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
        else:
            st.warning(
                f"SymSpell dictionary file '{dict_path}' not found. Spell correction disabled. "
                "You can download it from https://github.com/wolfgarbe/SymSpell/blob/master/SymSpell/frequency_dictionary_en_82_765.txt"
            )
            self.sym_spell = None

        self.ABBREVIATIONS = {
            "u": "you", "r": "are", "ur": "your", "dept": "department",
            "uni": "university", "admis": "admission", "fac": "faculty",
            "lect": "lecturer", "prof": "professor", "asap": "as soon as possible"
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
        if os.path.exists(config.DOCS_DIR) and os.listdir(config.DOCS_DIR):
            try:
                documents = SimpleDirectoryReader(config.DOCS_DIR).load_data()
                llm = LlamaOpenAI(model=config.LLM_MODEL, temperature=0.2)
                service_context = ServiceContext.from_defaults(llm=llm)
                self.rag_index = VectorStoreIndex.from_documents(documents, service_context=service_context)
            except Exception as e:
                st.warning(f"Failed to build RAG index: {str(e)}")
                self.rag_index = None
        else:
            self.rag_index = None
            st.warning(
                f"RAG documents directory '{config.DOCS_DIR}' is empty or missing. Please add docs for knowledge retrieval."
            )

    def preprocess_input(self, query):
        original_query = query
        corrected = None
        expanded_words = []

        if self.sym_spell:
            suggestions = self.sym_spell.lookup_compound(query, max_edit_distance=2)
            if suggestions and suggestions[0].term.lower() != query.lower():
                corrected = suggestions[0].term
                query = corrected

        words = query.split()
        new_words = []
        for w in words:
            if w.lower() in self.ABBREVIATIONS:
                expanded_words.append((w, self.ABBREVIATIONS[w.lower()]))
                new_words.append(self.ABBREVIATIONS[w.lower()])
            else:
                new_words.append(w)
        query = " ".join(new_words).lower().strip()

        if self.db and (corrected or expanded_words):
            event_data = {
                "original": original_query,
                "corrected": corrected,
                "expansions": expanded_words,
                "final_query": query
            }
            self.db.log_analytics("typo_abbreviation", event_data)

        return query

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
            try:
                engine = self.rag_index.as_query_engine()
                return str(engine.query(query))
            except Exception as e:
                st.warning(f"RAG query failed: {str(e)}")
        return None

    def detect_small_talk(self, query):
        small_talk_map = {
            "hi": ["Hello!", "Hi there!", "Hey!"],
            "hello": ["Hello!", "Hi!", "Hey!"],
            "hey": ["Hey!", "Hello!"],
            "good morning": ["Good morning! How can I help?"],
            "good afternoon": ["Good afternoon! What can I do for you?"],
            "good evening": ["Good evening! How may I assist?"],
            "how are you": ["I'm great, thanks for asking! How about you?"],
            "bye": ["Goodbye! Have a great day!", "See you later!"],
            "thanks": ["You're welcome!", "Glad I could help!"],
            "thank you": ["You're welcome!", "Anytime!"]
        }
        for phrase, responses in small_talk_map.items():
            if phrase in query.lower():
                user_name = st.session_state.get("user_name", "there")
                return f"{random.choice(responses)} {user_name}"
        return None

    def get_response(self, query, session_history=None):
        query = self.preprocess_input(query)

        small_talk = self.detect_small_talk(query)
        if small_talk:
            return small_talk, "small_talk"

        score, answer, match = self.get_best_match(query)
        if score > config.SIMILARITY_THRESHOLD:
            return answer, "qa_match"

        rag_answer = self.query_rag(query)
        if rag_answer and len(rag_answer) > 20:
            return rag_answer, "rag_system"

        context = ""
        if session_history:
            last_pairs = session_history[-3:]
            context = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in last_pairs if a])

        sentiment = TextBlob(query).sentiment.polarity if query else 0
        tone = ""
        if sentiment < -0.3:
            tone = "I'm sorry to hear that. I'll do my best to help you."
        elif sentiment > 0.3:
            tone = "That's great! How can I assist you further?"

        user_name = st.session_state.get("user_name", "User")
        user_dept = st.session_state.get("user_dept", "General")

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful, friendly, and slightly humorous assistant for Crescent University. "
                    "Respond clearly and politely."
                )
            },
            {
                "role": "user",
                "content": (
                    f"{tone}\n"
                    f"User: {user_name}\n"
                    f"Department: {user_dept}\n"
                    f"Conversation history:\n{context}\n\nUser: {query}"
                )
            }
        ]

        try:
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=prompt_messages,
                temperature=0.4,
                max_tokens=600,
                stream=True
            )
            return response, "llm_stream"
        except Exception as e:
            if score < config.LOW_CONFIDENCE_THRESHOLD:
                return "I’m not quite sure I understand. Could you please clarify or rephrase your question?", "clarification"
            else:
                fallback_msg = answer or "Sorry, I couldn't find an answer to that."
                return fallback_msg, "qa_match_fallback"

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
            st.session_state.session_id = str(uuid.uuid4())  # Unique session ID
        if 'user_name' not in st.session_state:
            st.session_state.user_name = ""
        if 'user_dept' not in st.session_state:
            st.session_state.user_dept = "General"

    def show(self):
        st.title(":sparkles: Crescent University Assistant Pro")

        # Ask for user name if not set
        if not st.session_state.user_name:
            name = st.text_input("Hi! What's your name?", key="user_name_input")
            if name:
                st.session_state.user_name = name.strip()
                st.success(f"Nice to meet you, {st.session_state.user_name}!")
            st.stop()

        # Ask for user department if not set or allow change
        dept = st.selectbox(
            "Select your department:",
            options=[
                "General",
                "Computer Science",
                "Chemical Science",
                "Mechanical Engineering",
                "Business Administration",
                "Law",
                # Add more departments as needed
            ],
            index=0,
            key="user_dept_select"
        )
        st.session_state.user_dept = dept

        # Add Reset Chat button
        if st.button("🔄 Reset Chat"):
            st.session_state.history = []
            st.experimental_rerun()

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
                        # Removed typing cursor "|" for cleaner look
                        message_placeholder.markdown(full_response)
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
            st.session_state.user_dept,
            query,
            full_response
        )
        # Log sentiment for analytics too
        sentiment_score = TextBlob(query).sentiment.polarity
        self.db.log_analytics("response", {
            "query": query,
            "source": source,
            "length": len(full_response),
            "sentiment": sentiment_score,
            "user_dept": st.session_state.user_dept
        })

# --- Main App ---
def main():
    db = DatabaseManager()
    ai = AIService(db=db)
    chat = ChatInterface(ai, db)
    chat.show()

if __name__ == "__main__":
    main()
