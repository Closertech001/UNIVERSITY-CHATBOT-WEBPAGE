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
import pkg_resources

# --- Page Config ---
st.set_page_config(page_title="🎓 Crescent Uni Assistant Pro", layout="wide")

# --- Config ---
class Config:
    def __init__(self):
        self.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        self.SIMILARITY_THRESHOLD = 0.70
        self.LLM_MODEL = "gpt-3.5-turbo"
        self.EMBEDDING_MODEL = "all-MiniLM-L12-v2"
        self.DOCS_DIR = "./university_docs/"
        self.ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin123")

config = Config()
openai.api_key = config.OPENAI_API_KEY

# --- Abbreviations & Synonyms ---
ABBREVIATIONS = {
    "u": "you", "r": "are", "ur": "your", "cn": "can", "pls": "please",
    "dept": "department", "uni": "university", "info": "information"
}
SYNONYMS = {
    "fees": "tuition fees", "tuition": "tuition fees", "school fees": "tuition fees",
    "course": "program", "programs": "courses", "faculty": "department"
}

# --- SymSpell Setup for spell checking ---
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# --- Database Manager ---
class DatabaseManager:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname=st.secrets.get("DB_NAME", "chatbot"),
            user=st.secrets.get("DB_USER", "postgres"),
            password=st.secrets.get("DB_PASSWORD", ""),
            host=st.secrets.get("DB_HOST", "localhost")
        )
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS unanswered (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT,
                    user_name TEXT,
                    user_dept TEXT,
                    question TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source TEXT
                )
            """)
            self.conn.commit()

    def store_conversation(self, session_id, user_name, user_dept, question, answer):
        with self.conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO memory (session_id, user_name, user_dept, question, answer)
                VALUES (%s, %s, %s, %s, %s)
            """, (session_id, user_name, user_dept, question, answer))
            self.conn.commit()

    def log_analytics(self, event_type, event_data):
        with self.conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO analytics (event_type, event_data)
                VALUES (%s, %s)
            """, (event_type, json.dumps(event_data)))
            self.conn.commit()

    def log_unanswered(self, session_id, user_name, user_dept, question, source):
        with self.conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO unanswered (session_id, user_name, user_dept, question, source)
                VALUES (%s, %s, %s, %s, %s)
            """, (session_id, user_name, user_dept, question, source))
            self.conn.commit()

# --- AI Service ---
def load_embedding_model(name):
    return SentenceTransformer(name)

def detect_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

class AIService:
    def __init__(self):
        self.embedding_model = load_embedding_model(config.EMBEDDING_MODEL)
        self.llm_model = config.LLM_MODEL
        self.questions = []
        self.answers = []
        self.qa_embeddings = None
        self.rag_index = None
        self.load_qa_data()
        self.build_rag_index()

    def load_qa_data(self):
        try:
            with open("qa_data.json", "r") as f:
                data = json.load(f)
                self.questions = [item['question'] for item in data]
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

    def preprocess_input(self, query):
        # Lowercase and strip
        query = query.lower().strip()

        # Replace abbreviations
        words = query.split()
        words = [ABBREVIATIONS.get(w, w) for w in words]

        # Replace synonyms
        words = [SYNONYMS.get(w, w) for w in words]

        # Join back
        query = " ".join(words)

        # Spell correction using SymSpell
        suggestions = sym_spell.lookup_compound(query, max_edit_distance=2)
        if suggestions:
            query = suggestions[0].term

        return query

    def get_best_match(self, query):
        if not self.qa_embeddings:
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
                return "Hello there! 😊 How can I assist you today?"
        return None

    def get_response(self, query, session_history=None):
        # Clarification logic
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
        # Clarification triggered if special "clarification" question detected
        if match == "clarification":
            st.session_state.clarification_options = answer.split('\n')[1:-2]
            return answer, "clarification"

        if score > config.SIMILARITY_THRESHOLD:
            return answer, "qa_match"

        rag_answer = self.query_rag(query)
        if rag_answer and len(rag_answer) > 20:
            return rag_answer, "rag_system"

        # Website-based shortcuts
        site_links = {
            "admission": "https://cuab.edu.ng/admission",
            "school fees": "https://cuab.edu.ng/tuition-fees/",
            "calendar": "https://cuab.edu.ng/academic-calendar",
            "contact": "https://cuab.edu.ng/contact",
            "programs": "https://cuab.edu.ng/undergraduate-programmes"
        }
        for keyword, link in site_links.items():
            if keyword in query.lower():
                return f"You can learn more about **{keyword}** [here]({link}).", "cuab_link"

        # GPT fallback with context & sentiment-aware tone
        context = "\n".join([f"Q: {q}\nA: {a}" for q, a in (session_history[-3:] if session_history else [])])
        sentiment = detect_sentiment(query)
        if sentiment == "negative":
            tone_prefix = "I'm really sorry you're having trouble. Let’s see how I can help 💙.\n"
        elif sentiment == "positive":
            tone_prefix = "Great to hear from you! 😄\n"
        else:
            tone_prefix = ""

        prompt = f"{tone_prefix}You're a thoughtful and smart university assistant. Think step-by-step and respond in a clear, friendly way.\n{context}\n\nUser: {query}\nAssistant:"

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
            # Log unanswered queries for review
            st.session_state.db.log_unanswered(
                st.session_state.session_id,
                st.session_state.user_name,
                st.session_state.user_dept,
                query,
                "error"
            )
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
        if 'typing' not in st.session_state:
            st.session_state.typing = False
        if 'user_name' not in st.session_state:
            st.session_state.user_name = "Anonymous"
        if 'user_dept' not in st.session_state:
            st.session_state.user_dept = "General"
        if 'db' not in st.session_state:
            st.session_state.db = self.db

    def show(self):
        st.title(":sparkles: Crescent University Assistant Pro")

        # User profile input
        with st.sidebar.expander("Your Profile"):
            user_name = st.text_input("Your Name", st.session_state.user_name)
            user_dept = st.text_input("Your Department", st.session_state.user_dept)
            if user_name != st.session_state.user_name:
                st.session_state.user_name = user_name
            if user_dept != st.session_state.user_dept:
                st.session_state.user_dept = user_dept

        if st.session_state.typing:
            st.info("🤖 Assistant is typing...")

        prompt = st.chat_input("Ask me anything about the university...")
        if prompt and not st.session_state.typing:
            st.session_state.history.append((prompt, ""))
            self.respond(prompt)

        for i, (q, a) in enumerate(st.session_state.history):
            # User messages right aligned, blue bubble
            st.markdown(f'<div style="text-align:right; background:#daf1fc; border-radius:10px; padding:8px; margin:4px 0;">{q}</div>', unsafe_allow_html=True)
            if a:
                # Assistant messages left aligned, light gray bubble with markdown
                st.markdown(f'<div style="text-align:left; background:#f1f1f1; border-radius:10px; padding:8px; margin:4px 0;">{a}</div>', unsafe_allow_html=True)

    def respond(self, query):
        st.session_state.typing = True
        self.show()  # Show typing indicator

        response, source = self.ai.get_response(query, st.session_state.history)
        time.sleep(1.2)  # Simulate typing delay

        st.session_state.history[-1] = (query, response)
        self.db.store_conversation(
            st.session_state.session_id, st.session_state.user_name, st.session_state.user_dept, query, response
        )
        self.db.log_analytics("response", {
            "query": query, "source": source, "length": len(response)
        })

        st.session_state.typing = False
        self.show()

# --- Main App ---
def main():
    db = DatabaseManager()
    ai = AIService()
    chat = ChatInterface(ai, db)
    chat.show()

if __name__ == "__main__":
    main()
