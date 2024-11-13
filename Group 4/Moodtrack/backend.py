# backend.py
import os
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv

def initialize_model():
    load_dotenv(find_dotenv())
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    return genai.GenerativeModel()

def handle_user_query(msg, chatbot):
    chatbot += [[msg, None]]
    return '', chatbot

def generate_chatbot(chatbot: list[list[str, str]]) -> list[list[str, str]]:
    formatted_chatbot = []
    
    if len(chatbot) == 0:
        return formatted_chatbot
    
    for ch in chatbot:
        # Ensure that each item has both user query and response (avoid index error)
        if len(ch) == 2 and ch[0] and ch[1] is not None:
            formatted_chatbot.append(
                {
                    "role": "user",
                    "parts": [ch[0]]
                }
            )
            formatted_chatbot.append(
                {
                    "role": "model",
                    "parts": [ch[1]]
                }
            )
        else:
            print(f"Skipping invalid chatbot entry: {ch}")  # Debugging invalid entries
    
    return formatted_chatbot


def handle_gemini_response(chatbot, model):
    # Ensure chatbot list is not empty
    if not chatbot or len(chatbot[-1]) < 2:
        print("Chatbot list is empty or improperly formatted.")
        return chatbot
    
    # Extract the user query (the first element of the last conversation)
    query = chatbot[-1][0]
    
    # Ensure the query is valid (not None or empty)
    if not query:
        print("Invalid query. Skipping Gemini response generation.")
        chatbot[-1][1] = "No input provided."
        return chatbot
    
    # Generate history for the conversation
    formatted_chatbot = generate_chatbot(chatbot[:-1])
    
    # Start chat and send message to the model
    try:
        chat = model.start_chat(history=formatted_chatbot)
        response = chat.send_message(query)
        chatbot[-1][1] = response.text
    except Exception as e:
        print(f"Error in Gemini response: {e}")
        if len(chatbot[-1]) == 2:  # Ensure there's a spot for the response
            chatbot[-1][1] = "Error in generating response. Please try again."
        else:
            chatbot.append([query, "Error in generating response. Please try again."])

    return chatbot


