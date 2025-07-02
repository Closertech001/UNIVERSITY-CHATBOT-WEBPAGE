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

# --- Database Manager ---
class DatabaseManager:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname=st.secrets.get("DB_NAME", "chatbot"),
            user=st.secrets.get("DB_USER", "postgres"),
            password=st.secrets.get("DB_PASSWORD", ""),
            host=st.secrets.get("DB_HOST", "localhost")
        )

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

# --- AI Service ---
def load_embedding_model(name):
    return SentenceTransformer(name)

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
        return query.lower().strip()

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
        query = self.preprocess_input(query)

        # Detect small talk first
        small_talk = self.detect_small_talk(query)
        if small_talk:
            return small_talk, "small_talk"

        score, answer, match = self.get_best_match(query)
        if score > config.SIMILARITY_THRESHOLD:
            return answer, "qa_match"

        rag_answer = self.query_rag(query)
        if rag_answer and len(rag_answer) > 20:
            return rag_answer, "rag_system"

        context = "\n".join([f"Q: {q}\nA: {a}" for q, a in (session_history[-3:] if session_history else [])])
        prompt = f"You are a helpful, friendly, and slightly humorous assistant for Crescent University.\nAlways keep responses clear, short, and encouraging.\n\n{context}\n\nUser: {query}\nAssistant:"

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

        if source == "llm_stream":
            try:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    for chunk in response:
                        token = chunk.choices[0].delta.get("content", "")
                        full_response += token
                        message_placeholder.markdown(full_response + "|")
                        time.sleep(0.03)
            except Exception as e:
                full_response = f"[Stream Error: {str(e)}]"
                st.error(full_response)
        else:
            full_response = response
            with st.chat_message("assistant"):
                st.markdown(full_response)

        st.session_state.history[-1] = (query, full_response)
        self.db.store_conversation(
            st.session_state.session_id, "Anonymous", "General", query, full_response
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
