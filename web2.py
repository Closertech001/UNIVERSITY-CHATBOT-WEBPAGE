import streamlit as st
import os
import json
import re
import faiss
import openai
import logging
from sentence_transformers import SentenceTransformer
from symspellpy import SymSpell, Verbosity
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(filename='gpt_fallback_logs.txt', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.65"))

# Validate OpenAI API key
if not openai.api_key:
    st.error("OpenAI API key not found. Please set it in the .env file.")
    st.stop()

# Initialize SentenceTransformer model
@st.cache_resource
def load_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Failed to load SentenceTransformer model: {e}")
        st.stop()

model = load_model()

# Load SymSpell for spell correction
@st.cache_resource
def load_symspell():
    try:
        sym = SymSpell(max_dictionary_edit_distance=2)
        dict_path = "data/frequency_dictionary_en_82_765.txt"
        if not os.path.exists(dict_path):
            st.warning("Dictionary file not found. Spell correction disabled.")
            return None
        sym.load_dictionary(dict_path, term_index=0, count_index=1)
        return sym
    except Exception as e:
        st.error(f"Failed to load SymSpell dictionary: {e}")
        st.stop()

symspell = load_symspell()

# Load dataset and FAISS index
@st.cache_resource
def load_chunks_index():
    try:
        if not os.path.exists("data/qa_data_cleaned.json"):
            st.error("Dataset file not found. Please ensure 'qa_data_cleaned.json' exists.")
            st.stop()
        with open("data/qa_data_cleaned.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
        # Validate dataset structure
        for q in chunks:
            if "question" not in q or "answer" not in q:
                st.error("Invalid dataset format. Each entry must have 'question' and 'answer' fields.")
                st.stop()
        questions = [q["question"] for q in chunks]
        embeddings = model.encode(questions, convert_to_numpy=True).astype("float32")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return chunks, index, questions
    except Exception as e:
        st.error(f"Failed to load dataset or index: {e}")
        st.stop()

chunks, index, questions = load_chunks_index()

# Input normalization
def normalize_input(user_input):
    if not user_input:
        return ""
    # Dictionary for synonyms and abbreviations
    replacements = {
        "uni": "university",
        "courses": "subjects",
        "cgpa": "cumulative grade point average",
        "dept": "department",
        "sem": "semester"
    }
    
    user_input = sanitize_input(user_input.lower())
    
    # Replace synonyms and abbreviations
    for key, value in replacements.items():
        user_input = user_input.replace(key, value)
    
    # Spell correction
    if symspell:
        suggestions = symspell.lookup_compound(user_input, max_edit_distance=2)
        if suggestions:
            user_input = suggestions[0].term
    
    return user_input

# Input sanitization
def sanitize_input(text):
    return re.sub(r'[<>]', '', text.strip())

# Detect greetings
def detect_greeting(user_input):
    greetings = r"\b(hi|hello|hey|good\s+(morning|afternoon|evening))\b"
    return re.search(greetings, user_input.lower()) is not None

# Query GPT with fallback
def ask_gpt(query, chat_history, max_retries=2):
    context = "\n".join([f"User: {q}\nBot: {a}" for q, a, _ in chat_history[-5:]])
    prompt = f"Context:\n{context}\n\nQuery: {query}\n\nAnswer as a knowledgeable assistant for Crescent University."
    
    for attempt, model in enumerate(["gpt-4", "gpt-3.5-turbo"]):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are CrescentBot, a helpful assistant for Crescent University."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300
            )
            return response.choices[0].message.content.strip(), model
        except Exception as e:
            logging.error(f"Error with {model}: {str(e)}")
            if attempt == max_retries - 1:
                st.warning("Both GPT-4 and GPT-3.5 are unavailable. Check your connection or try again later.")
                return "I'm currently unable to fetch a detailed answer. Please try again later, or rephrase your question for better results.", "fallback"
    
    return "I'm currently unable to fetch a detailed answer. Please try again later, or rephrase your question for better results.", "fallback"

# Streamlit UI
st.title("Crescent University Chatbot")
st.write("Ask about courses, departments, or anything related to Crescent University!")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input field
user_input = st.text_input("Your Question:", placeholder="e.g. What courses are offered in 200 level Law?")
if user_input:
    # Sanitize and normalize input
    norm_query = normalize_input(user_input)
    
    # Handle greetings
    if detect_greeting(user_input):
        response = "Hello! How can I assist you with Crescent University today?"
        source = "greeting"
    else:
        # Semantic search
        query_embedding = model.encode([norm_query], convert_to_numpy=True).astype("float32")
        distances, indices = index.search(query_embedding, 1)
        score = 1 - distances[0][0] / 2  # Convert L2 distance to similarity
        
        if score >= SEMANTIC_THRESHOLD:
            response = chunks[indices[0][0]]["answer"]
            source = "local"
        else:
            response, source = ask_gpt(norm_query, st.session_state.chat_history)
    
    # Append to chat history (limit to 20 messages)
    st.session_state.chat_history.append((user_input, response, source))
    if len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[-20:]

# Display chat history
for question, answer, source in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(f"CrescentBot ({source.capitalize()}): {answer}")
