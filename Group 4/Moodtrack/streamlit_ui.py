import streamlit as st
from backend import handle_gemini_response, initialize_model



# Initialize session state for chatbot
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = []

model = initialize_model()  # Make sure this function works properly

st.title("Chatbot")

# User input
user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        # Add user message to chatbot history
        st.session_state.chatbot.append([user_input, None])  # User message, placeholder for bot response
        
        # Generate bot response
        st.session_state.chatbot = handle_gemini_response(st.session_state.chatbot, model)

# Display the chat history
for message_pair in st.session_state.chatbot:
    if message_pair:
        user_message = message_pair[0]
        bot_response = message_pair[1] if len(message_pair) > 1 else "Waiting for response..."
        st.write(f"You: {user_message}")
        st.write(f"Bot: {bot_response}")
    else:
        st.write("No messages.")
