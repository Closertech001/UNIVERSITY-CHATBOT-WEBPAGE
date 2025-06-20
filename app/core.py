import re
import torch
import random
import openai
import time
from sentence_transformers import util
import streamlit as st
from .normalizer import normalize_text, extract_course_codes
from .context_manager import detect_emotion, detect_department, update_conversation_context
from .data_loader import qa_data, qa_embeddings, model, SYSTEM_PROMPT

def is_greeting(text):
    return any(word in text.lower() for word in ["hello", "hi", "hey", "good morning", "good evening", "good day"])

def get_gpt_answer(user_query):
    try:
        ctx = st.session_state.get("conversation_context", {})
        context_clues = []

        # Inject memory of department/level
        if ctx.get("current_department"):
            context_clues.append(f"The user is asking about the {ctx['current_department']} department.")
        if ctx.get("current_level"):
            context_clues.append(f"They are referring to {ctx['current_level']} level.")
        context_summary = " ".join(context_clues)

        # Build GPT prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context_summary}]

        # Add chat history (last 3 turns for short-term memory)
        messages += [
            {"role": "user" if i % 2 == 0 else "assistant", "content": msg["user"] if i % 2 == 0 else msg["bot"]}
            for i, msg in enumerate(st.session_state.chat_history[-4:])
        ]

        # Add current query
        messages.append({"role": "user", "content": user_query})

        # Get GPT completion
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            presence_penalty=0.5,
            frequency_penalty=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Hmm, I’m having a hard time thinking right now 😕. Could you repeat that another way?"

def search_answer(user_query, threshold=0.5):
    processed_query = normalize_text(user_query)
    department = detect_department(processed_query)
    course_codes = extract_course_codes(processed_query)

    for qa in qa_data:
        norm_q = normalize_text(qa["question"])
        if processed_query == norm_q or (department and department in norm_q and processed_query in norm_q):
            return qa["answer"]
        if course_codes and any(code in norm_q for code in course_codes) and processed_query in norm_q:
            return qa["answer"]

    query_embed = model.encode(processed_query, convert_to_tensor=True)
    scores = util.cos_sim(query_embed, qa_embeddings)[0]

    for i, qa in enumerate(qa_data):
        norm_q = normalize_text(qa["question"])
        if department and department in norm_q:
            scores[i] += 0.2
        if course_codes and any(code in norm_q for code in course_codes):
            scores[i] += 0.1

    best_idx = torch.argmax(scores).item()
    return qa_data[best_idx]["answer"] if scores[best_idx] > threshold else None

def format_response(response, emotion):
    if emotion == "negative":
        response = f"I'm really sorry to hear that. 😔 Let me help - {response}"
    elif emotion == "positive":
        response = f"Great! 😊 {response}"
    return response.replace("The answer is", "Here's what I found").replace("According to our records", "From what I know")

def stream_response(response):
    placeholder = st.empty()
    output = ""
    parts = re.split(r'(?<=[,.!?])\s+', response)
    for part in parts:
        output += part + " "
        time.sleep(0.05 * len(part.split()))
        placeholder.markdown(output + "▌")
    placeholder.markdown(output)
    return output

def get_response(user_query):
    from .context_manager import enrich_followup_query

    # If follow-up, enrich question using stored department
    enriched_query = enrich_followup_query(user_query)

    if is_greeting(user_query):
        return "Hey there! 😊 How can I help you today?"

    answer = search_answer(enriched_query)
    if not answer:
        answer = get_gpt_answer(enriched_query)

    emotion = detect_emotion(user_query)
    update_conversation_context(enriched_query, answer)
    return format_response(answer, emotion)

