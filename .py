import streamlit as st
import json
import time
import os
import openai
from sentence_transformers import SentenceTransformer, util
from symspellpy.symspellpy import SymSpell, Verbosity
from textblob import TextBlob
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI as LlamaOpenAI

# -------------------
# Configuration
# -------------------
class Config:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    SIMILARITY_THRESHOLD = 0.7
    CLARIFICATION_THRESHOLD = 0.4
    LLM_MODEL = "gpt-3.5-turbo"
    EMBEDDING_MODEL = "all-MiniLM-L12-v2"
    DOCS_DIR = "./university_docs/"
    SYMSPELL_DICT = "frequency_dictionary_en_82_765.txt"

config = Config()
openai.api_key = config.OPENAI_API_KEY

# -------------------
# Load SymSpell
# -------------------
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
if not sym_spell.load_dictionary(config.SYMSPELL_DICT, term_index=0, count_index=1):
    st.error("SymSpell dictionary file not found!")

ABBREVIATIONS = {
    "u": "you",
    "r": "are",
    "ur": "your",
    "dept": "department",
    "uni": "university",
    # Add more as needed
}

# -------------------
# AI Service
# -------------------
class AIService:
    def __init__(self):
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.questions, self.answers = [], []
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
            st.error(f"Failed to load QA data: {e}")

    def build_rag_index(self):
        if os.path.exists(config.DOCS_DIR):
            documents = SimpleDirectoryReader(config.DOCS_DIR).load_data()
            llm = LlamaOpenAI(model=config.LLM_MODEL, temperature=0.2)
            service_context = ServiceContext.from_defaults(llm=llm)
            self.rag_index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    def preprocess_input(self, query):
        # Spell correction
        suggestions = sym_spell.lookup_compound(query, max_edit_distance=2)
        if suggestions:
            query = suggestions[0].term
        # Expand abbreviations
        words = query.split()
        words = [ABBREVIATIONS.get(w.lower(), w) for w in words]
        return " ".join(words).lower().strip()

    def detect_small_talk(self, query):
        small_talk_triggers = ["hi", "hello", "hey", "good morning", "good afternoon", "how are you"]
        for phrase in small_talk_triggers:
            if phrase in query.lower():
                user_name = st.session_state.get("user_name", "there")
                return f"Hello {user_name}! 😊 How can I assist you today?"
        return None

    def get_best_match(self, query):
        if not self.qa_embeddings:
            return 0, None
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.qa_embeddings)[0]
        best_score = float(scores.max())
        best_index = int(scores.argmax())
        return best_score, self.answers[best_index]

    def query_rag(self, query):
        if self.rag_index:
            engine = self.rag_index.as_query_engine()
            return str(engine.query(query))
        return None

    def get_response(self, query, session_history=None):
        query = self.preprocess_input(query)

        # Check small talk
        small_talk = self.detect_small_talk(query)
        if small_talk:
            return small_talk, "small_talk"

        score, answer = self.get_best_match(query)
        if score > config.SIMILARITY_THRESHOLD:
            return answer, "qa_match"

        rag_answer = self.query_rag(query)
        if rag_answer and len(rag_answer) > 20:
            return rag_answer, "rag_system"

        if score < config.CLARIFICATION_THRESHOLD:
            return "I’m not sure I understand. Could you please clarify your question?", "clarification"

        # Sentiment analysis
        sentiment = TextBlob(query).sentiment.polarity
        tone_intro = ""
        if sentiment < -0.3:
            tone_intro = "I'm sorry to hear that. I'll do my best to help you.\n"
        elif sentiment > 0.3:
            tone_intro = "That's great! How can I assist you further?\n"

        # Prepare multi-turn context (last 3 pairs)
        context = ""
        if session_history:
            last_pairs = session_history[-3:]
            context = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in last_pairs if a])

        prompt = (
            f"You are a friendly, helpful, and empathetic assistant for Crescent University.\n"
            f"{tone_intro}"
            f"Use the following conversation history for context:\n{context}\n"
            f"User: {query}\nAssistant:"
        )

        try:
            response = openai.ChatCompletion.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=600,
                stream=True,
            )
            return response, "llm_stream"
        except Exception as e:
            return f"[Error generating response: {e}]", "error"

# -------------------
# Chat Interface
# -------------------
class ChatInterface:
    def __init__(self, ai_service):
        self.ai = ai_service
        self.init_session()

    def init_session(self):
        if "history" not in st.session_state:
            st.session_state.history = []
        if "user_name" not in st.session_state:
            st.session_state.user_name = ""

    def show(self):
        st.title("🎓 Crescent University Assistant")

        if not st.session_state.user_name:
            name = st.text_input("Hi! What's your name?")
            if name:
                st.session_state.user_name = name
                st.success(f"Welcome, {name}! Ask me anything about the university.")
            return

        prompt = st.chat_input("Ask me anything about Crescent University...")
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
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.02)
            except Exception as e:
                full_response = f"[Stream Error: {e}]"
                st.error(full_response)
        else:
            full_response = response
            with st.chat_message("assistant"):
                st.markdown(full_response)

        st.session_state.history[-1] = (query, full_response)

# -------------------
# Main
# -------------------
def main():
    st.set_page_config(page_title="Crescent University Assistant", layout="wide")
    ai_service = AIService()
    chat_interface = ChatInterface(ai_service)
    chat_interface.show()

if __name__ == "__main__":
    main()
