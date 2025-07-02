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
import difflib  # For dynamic typo correction

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

SYNONYM_CANDIDATES = [
    "school fees", "tuition", "fees", "accommodation", "hostel", "ict department", "it department",
    "academic staff", "lecturers", "non-academic staff", "admin staff", "programs", "courses",
    "computer science", "comp sci", "biochemistry", "bio"
]

class AIService:
    def __init__(self):
        self.embedding_model = load_embedding_model(config.EMBEDDING_MODEL)
        self.llm_model = config.LLM_MODEL
        self.questions = []
        self.answers = []
        self.qa_embeddings = None
        self.rag_index = None
        self.synonym_embeddings = self.embedding_model.encode(SYNONYM_CANDIDATES, convert_to_tensor=True)
        self.load_qa_data()
        self.build_rag_index()

    def correct_spelling(self, text):
        corrected_words = []
        for word in text.split():
            close_matches = difflib.get_close_matches(word, SYNONYM_CANDIDATES, n=1, cutoff=0.85)
            corrected_words.append(close_matches[0] if close_matches else word)
        return " ".join(corrected_words)

    def semantic_replace(self, query):
        tokens = query.split()
        updated = []
        for token in tokens:
            token_vec = self.embedding_model.encode(token, convert_to_tensor=True)
            similarities = util.cos_sim(token_vec, self.synonym_embeddings)[0]
            best_idx = int(similarities.argmax())
            best_score = float(similarities[best_idx])
            updated.append(SYNONYM_CANDIDATES[best_idx] if best_score > 0.8 else token)
        return " ".join(updated)

    def preprocess_input(self, query):
        query = query.lower().strip()
        query = self.correct_spelling(query)
        query = self.semantic_replace(query)
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

    def get_best_match(self, query):
        if self.qa_embeddings is None:
            return 0, None, None
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.qa_embeddings)[0]
        sorted_scores = sorted([(i, float(s)) for i, s in enumerate(scores)], key=lambda x: -x[1])

        if sorted_scores[0][1] >= config.SIMILARITY_THRESHOLD:
            top_matches = sorted_scores[:3]  # take top 3 if possible
            if top_matches[1][1] > 0.6:
                options = [self.questions[i] for i, _ in top_matches]
                clarification_prompt = "I found multiple similar questions. Did you mean:\n"
                for idx, opt in enumerate(options):
                    clarification_prompt += f"{idx+1}) {opt}\n"
                clarification_prompt += "\nPlease respond with the number."
                return 0.66, clarification_prompt, "clarification"

            best_index = sorted_scores[0][0]
            return sorted_scores[0][1], self.answers[best_index], self.questions[best_index]
        return 0, None, None

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

        if query.isdigit() and 'clarification_options' in st.session_state:
            options = st.session_state.pop('clarification_options', [])
            try:
                selected_idx = int(query) - 1
                if 0 <= selected_idx < len(options):
                    query = options[selected_idx]
            except:
                return "Sorry, I didn't understand that selection.", "clarification_error"

        small_talk = self.detect_small_talk(query)
        if small_talk:
            return small_talk, "small_talk"

        score, answer, match = self.get_best_match(query)
        if match == "clarification":
            st.session_state.clarification_options = answer.split('
')[1:-2]
            return answer, "clarification"

        if score > config.SIMILARITY_THRESHOLD:
            return answer, "qa_match"

        rag_answer = self.query_rag(query)
        if rag_answer and len(rag_answer) > 20:
            return rag_answer, "rag_system"

        context = "
".join([f"Q: {q}
A: {a}" for q, a in (session_history[-3:] if session_history else [])])
        sentiment = detect_sentiment(query)
        if sentiment == "negative":
            tone_prefix = "I'm really sorry you're having trouble. Let’s see how I can help 💙.
"
        elif sentiment == "positive":
            tone_prefix = "Great to hear from you! 😄
"
        else:
            tone_prefix = ""

        prompt = f"{tone_prefix}You're a thoughtful and smart university assistant. Think step-by-step and respond in a clear, friendly way.
{context}

User: {query}
Assistant:"

        try:
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=600,
                stream=False
            )
            return response["choices"][0]["message"]["content"], "llm"
        except Exception as e:
            return f"[LLM Error: {str(e)}]", "error" small_talk, "small_talk"

        score, answer, match = self.get_best_match(query)
        if match == "clarification":
            return answer, "clarification"

        if score > config.SIMILARITY_THRESHOLD:
            return answer, "qa_match"

        rag_answer = self.query_rag(query)
        if rag_answer and len(rag_answer) > 20:
            return rag_answer, "rag_system"

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
    def __init__(self, ai, db):
        self.ai = ai
        self.db = db
        self.init_session()

    def init_session(self):
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(time.time())
        if 'user_name' not in st.session_state:
            st.session_state.user_name = ""
        if 'user_dept' not in st.session_state:
            st.session_state.user_dept = ""

    def show(self):
        st.title("✨ Crescent University Assistant Pro")

        if not st.session_state.user_name:
            st.session_state.user_name = st.text_input("👤 What is your name?")
            return
        if not st.session_state.user_dept:
            st.session_state.user_dept = st.text_input("🏫 What department are you in?")
            return

        previous = self.db.conn.execute("""
            SELECT question FROM memory 
            WHERE user_name = ? AND user_dept = ? 
            ORDER BY timestamp DESC LIMIT 1
        """, (st.session_state.user_name, st.session_state.user_dept)).fetchone()

        welcome_message = f"Welcome back, {st.session_state.user_name} from {st.session_state.user_dept}! 😊"
        if previous:
            welcome_message += f" Last time you asked: '{previous[0]}'"

                with st.chat_message("assistant"):
            st.markdown(welcome_message)

        # --- Proactive suggestions ---
        past_questions = self.db.conn.execute("""
            SELECT question FROM memory
            WHERE user_name = ? AND user_dept = ?
            ORDER BY timestamp DESC LIMIT 5
        """, (st.session_state.user_name, st.session_state.user_dept)).fetchall()

        if past_questions:
            related = [q[0] for q in past_questions if q[0] != previous[0]]
            if related:
                st.markdown("
#### 🔍 You might also be interested in:")
                for q in related[:2]:
                    st.markdown(f"- {q}")

        prompt = st.chat_input("Ask me anything...")
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
        response, source = self.ai.get_response(query, st.session_state.history)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.history[-1] = (query, response)
        self.db.store_conversation(
            st.session_state.session_id,
            st.session_state.user_name,
            st.session_state.user_dept,
            query,
            response
        )
        self.db.log_analytics("response", {
            "query": query,
            "source": source,
            "length": len(response)
        })

# --- Main App ---
def main():
    db = DatabaseManager()
    ai = AIService()
    chat = ChatInterface(ai, db)
    chat.show()

if __name__ == "__main__":
    main()
