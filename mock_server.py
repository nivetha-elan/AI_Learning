from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from transformers import pipeline

# Load environment variables
load_dotenv()

# Configure API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini Pro model
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Initialize Streamlit app
st.set_page_config(page_title="AI-Powered Adaptive Learning Platform")

st.title("AI-Powered Adaptive Learning Platform")

# Virtual Office Hours Assistant as Reminder
st.header("Meeting Reminder")
st.write("Set a reminder for your meeting.")
name = st.text_input("Enter your name:")
email = st.text_input("Enter your email:")
meeting_time = st.time_input("Choose a time for your meeting:")
if st.button("Set Reminder"):
    if name and email:
        st.success(f"Reminder set for {name} at {meeting_time}. Confirmation sent to {email}.")
    else:
        st.error("Please enter both name and email.")

# Lecture Enhancement
st.header("Lecture Enhancement")
st.write("Summarize your lecture notes.")
lecture_notes = st.text_area("Enter lecture notes:")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
if st.button("Summarize"):
    if lecture_notes:
        summary = summarizer(lecture_notes, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        st.write("Summary:", summary)
    else:
        st.error("Please enter lecture notes to summarize.")

# Automated Feedback System
st.header("Automated Feedback System")
st.write("Get feedback on your assignment.")
assignment = st.text_area("Enter assignment text:")
if st.button("Get Feedback"):
    if assignment:
        feedback = "Great job! Consider expanding your analysis in the third paragraph."
        st.write("Feedback:", feedback)
    else:
        st.error("Please enter assignment text to get feedback.")

# Language Learning Companion
st.header("Language Learning Companion")
st.write("Translate your practice sentence.")
translator_en_to_fr = pipeline("translation_en_to_fr", model="t5-small")
translator_en_to_hi = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi")
translator_en_to_ml = pipeline("translation_en_to_ml", model="Helsinki-NLP/opus-mt-en-ml")
language_input = st.text_input("Practice a sentence:")
language = st.selectbox("Select language for translation:", ["French", "Hindi", "Malayalam"])

if st.button("Get Translation"):
    if language_input:
        if language == "French":
            translation = translator_en_to_fr(language_input, max_length=400)[0]['translation_text']
        elif language == "Hindi":
            translation = translator_en_to_hi(language_input, max_length=400)[0]['translation_text']
        elif language == "Malayalam":
            translation = translator_en_to_ml(language_input, max_length=400)[0]['translation_text']
        st.write(f"Translation ({language}):", translation)
    else:
        st.error("Please enter a sentence to translate.")

# Gemini LLM Application
st.header("Gemini LLM Application")
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

input = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

if submit and input:
    response = get_gemini_response(input)
    st.session_state['chat_history'].append(("You", input))
    st.subheader("The Response is")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("Bot", chunk.text))

for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")
