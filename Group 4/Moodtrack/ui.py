# streamlit_ui.py
import streamlit as st
from backend import initialize_model, handle_user_query, handle_gemini_response

# Initialize the model
model = initialize_model()

# Streamlit UI setup
st.title("Mood Tracker Chatbot")

# Initialize chatbot list
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = []

# Create input field and button
user_input = st.text_input("You:", key="user_input")
submit_button = st.button("Send")

# If the button is clicked, handle the input
if submit_button and user_input:
    # Handle the user input and generate response
    chatbot = handle_user_query(user_input, st.session_state.chatbot)
    st.session_state.chatbot = handle_gemini_response(chatbot, model)

# Display chatbot conversation
if st.session_state.chatbot:
    for message_pair in st.session_state.chatbot:
        user_message, bot_response = message_pair
        st.write(f"**You**: {user_message}")
        st.write(f"**Bot**: {bot_response}")

