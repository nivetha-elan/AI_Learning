import os
import streamlit as st
import google.generativeai as genai
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
import pipeline

# Load environment variables
load_dotenv()

# Configure API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

def send_email_reminder(to_email, subject, body):
    from_email = os.getenv("EMAIL")
    password = os.getenv("EMAIL_PASSWORD")
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email
    
    with smtplib.SMTP_SSL('smtp.example.com', 465) as server:
        server.login(from_email, password)
        server.send_message(msg)

def generate_quiz_questions(topic, grade):
    query = f"Generate 10 quiz questions for {grade} students on the topic {topic} with 4 options and correct answer"
    response = get_gemini_response(query)
    content = "".join([chunk.text for chunk in response])
    questions = content.split("\n\n")  # assuming each question is separated by two newlines
    quiz = []
    for question in questions:
        parts = question.split("\n")  # split question and options
        if len(parts) >= 6:
            q = parts[0]
            options = parts[1:5]
            correct_answer = parts[5].replace("Correct answer: ", "").strip()
            quiz.append({"question": q, "options": options, "correct_answer": correct_answer})
    return quiz

# Initialize Streamlit app
st.set_page_config(page_title="AI-Powered Adaptive Learning Platform")

st.title("AI-Powered Adaptive Learning Platform")

# Tabs for different sections
section = st.sidebar.selectbox("Choose a section", ["Home", "Gemini LLM Application", "Meeting Reminder", "Lecture Enhancement", "Automated Feedback System", "Language Learning Companion", "Q&A Chatbot", "Automated Assignment Generator"])

if section == "Home":
    st.header("Welcome to the AI-Powered Adaptive Learning Platform")
    st.write("Choose a section from the sidebar to get started.")

elif section == "Gemini LLM Application":
    st.header("💬 Gemini LLM Application")
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

elif section == "Meeting Reminder":
    st.header("Meeting Reminder")
    st.write("Set a reminder for your meeting.")
    name = st.text_input("Enter your name:")
    email = st.text_input("Enter your email:")
    meeting_time = st.time_input("Choose a time for your meeting:")
    reminder_time = st.slider("Set reminder minutes before the meeting:", 5, 60, 15)

    if st.button("Set Reminder"):
        if name and email:
            meeting_datetime = datetime.combine(datetime.today(), meeting_time)
            reminder_datetime = meeting_datetime - timedelta(minutes=reminder_time)
            st.success(f"Reminder set for {name} at {meeting_time}. You'll receive a reminder {reminder_time} minutes before.")
            
            # Send an email reminder (in a real app, use a scheduling library or service)
            subject = "Meeting Reminder"
            body = f"Hi {name}, this is a reminder for your meeting at {meeting_time}."
            send_email_reminder(email, subject, body)
        else:
            st.error("Please enter both name and email.")

elif section == "Lecture Enhancement":
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

elif section == "Automated Feedback System":
    st.header("Automated Feedback System")
    st.write("Get feedback on your assignment.")
    assignment = st.text_area("Enter assignment text:")
    if st.button("Get Feedback"):
        if assignment:
            feedback = "Great job! Consider expanding your analysis in the third paragraph."
            st.write("Feedback:", feedback)
        else:
            st.error("Please enter assignment text to get feedback.")

elif section == "Language Learning Companion":
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

elif section == "Q&A Chatbot":
    st.header("Q&A Chatbot")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    user_input = st.text_input("Input: ", key="input")
    submit = st.button("Ask the question")

    if submit and user_input:
        response = get_gemini_response(user_input)
        st.session_state['chat_history'].append(("You", user_input))
        st.subheader("The Response is")
        for chunk in response:
            st.write(chunk.text)
            st.session_state['chat_history'].append(("Bot", chunk.text))

    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")

elif section == "Automated Assignment Generator":
    st.header("Automated Assignment Generator")
    st.write("Generate assignments based on the topic and grade.")
    topic = st.text_input("Enter the topic:")
    grade = st.selectbox("Select the grade level:", [
        "Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5",
        "Grade 6", "Grade 7", "Grade 8", "Grade 9", "Grade 10",
        "Grade 11", "Grade 12"
    ])

    if st.button("Generate Assignments"):
        if topic and grade:
            quiz = generate_quiz_questions(topic, grade)
            if 'quiz_data' not in st.session_state:
                st.session_state['quiz_data'] = {'questions': quiz, 'user_answers': [""]*10}
            st.write("### Quiz Questions:")
            with st.form("quiz_form"):
                for i, q in enumerate(st.session_state['quiz_data']['questions']):
                    st.write(f"Q{i+1}: {q['question']}")
                    st.session_state['quiz_data']['user_answers'][i] = st.radio(f"Options for Q{i+1}", options=q['options'], key=f"question_{i}_option")
                submit_quiz = st.form_submit_button("Submit Quiz")
            
            if submit_quiz:
                correct_answers = 0
                total_questions = len(st.session_state['quiz_data']['questions'])
                for i, q in enumerate(st.session_state['quiz_data']['questions']):
                    user_answer = st.session_state['quiz_data']['user_answers'][i]
                    if user_answer == q['correct_answer']:
                        correct_answers += 1
                
                score = (correct_answers / total_questions) * 100
                st.write(f"Your score is: {score}%")
                
                # Generate Performance Report
                performance_data = {
                    'Question': [f'Q{i+1}' for i in range(total_questions)],
                    'Your Answer': [st.session_state['quiz_data']['user_answers'][i] for i in range(total_questions)],
                    'Correct Answer': [q['correct_answer'] for q in st.session_state['quiz_data']['questions']]
                }
                df = pd.DataFrame(performance_data)
                st.write("### Performance Report:")
                st.write(df)
                
                # Plot Performance Chart
                fig, ax = plt.subplots()
                df_correct = df['Your Answer'] == df['Correct Answer']
                ax.bar(df['Question'], df_correct, color=['green' if val else 'red' for val in df_correct])
                ax.set_ylabel('Correct (1) / Incorrect (0)')
                ax.set_title('Quiz Performance')
                st.pyplot(fig)
        else:
            st.error("Please enter both topic and grade.")
