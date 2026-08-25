import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import streamlit as st
import sys
from loguru import logger
import time
import os
from pathlib import Path

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Repo-relative so this runs anywhere; override with DRIV_DATA if needed.
DATA_PATH = os.getenv(
    "DRIV_DATA",
    str(Path(__file__).resolve().parent / "Data cleaning" /
        "Cleaned_data_with_embeddings.csv"))

# Configure Google Generative AI API
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize tools for metadata extraction from user query
car_extractor = {
    'function_declarations': [
        {
            'name': 'get_required_cars',
            'description': 'Extract price and preferences for car recommendations.',
            'parameters': {
                'type_': 'OBJECT',
                'properties': {
                    'max_price': {'type_': 'NUMBER', 'description': 'The max budget for the car.'},
                    'semantic_search_query': {'type_': 'STRING', 'description': 'Car preferences (e.g., SUV with sunroof)'}
                },
                'required': ['max_price', 'semantic_search_query']
            }
        }
    ]
}

# Initialize the generative model with the car extraction tool
car_tool = genai.protos.Tool(car_extractor)
model = genai.GenerativeModel('gemini-1.5-flash', tools=[car_tool])

# Backend prompt to instruct the chatbot
backend_instruction = {
    'role': 'user',
    'parts': (
        "You are a car recommendation chatbot. You will receive a User Query and Retrieved Data. "
        "Use the Retrieved Data to answer the User Query. If no Retrieved Data is available, refer "
        "to the chat history to form your response."
    )
}

# Load Sentence Transformer and data
roberta_model = SentenceTransformer('all-roberta-large-v1')
df = pd.read_csv(DATA_PATH)
df['embeddings'] = df['embeddings'].apply(lambda x: np.fromstring(x[1:-1], sep=','))
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')
df = df.dropna(subset=['Price', 'Mileage'])

# Function to retrieve cars based on price range and best mileage with FAISS timing and cosine similarity
def get_required_cars(max_price, semantic_search_query, k=5):
    roberta_model = SentenceTransformer('all-roberta-large-v1')
    df = pd.read_csv(DATA_PATH)
    df['embeddings'] = df['embeddings'].apply(lambda x: np.fromstring(x[1:-1], sep=','))
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')
    df = df.dropna(subset=['Price', 'Mileage'])
    price_range = (max_price * 0.9, max_price * 1.1)  # Define price range within ±10%
    filtered_df = df[(df['Price'] >= price_range[0]) & (df['Price'] <= price_range[1])]
    filtered_df = filtered_df.sort_values(by='Mileage', ascending=False).drop_duplicates(subset=['Car'])

    # Start timing for FAISS indexing
    faiss_start = time.time()
    
    # Perform semantic search
    query_embedding = roberta_model.encode(semantic_search_query, convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    filtered_embeddings = np.vstack(filtered_df['embeddings'].values)
    index = faiss.IndexFlatIP(filtered_embeddings.shape[1])
    index.add(filtered_embeddings)
    distances, indices = index.search(np.array([query_embedding]), k)

    # End timing for FAISS indexing
    faiss_end = time.time()
    faiss_index_time = faiss_end - faiss_start

    # Print FAISS index time
    print(f"FAISS Index Time: {faiss_index_time:.4f} seconds")
    
    # Calculate cosine similarity with the top results
    cosine_similarities = [np.dot(query_embedding, filtered_embeddings[idx]) for idx in indices[0]]
    
    # Print cosine similarities for debugging
    print(f"Cosine Similarities: {cosine_similarities}")
    
    recommendations = filtered_df.iloc[indices[0]].drop_duplicates(subset=['Car'])
    recommendations['Cosine Similarity'] = cosine_similarities
    recommendations['FAISS Index Time'] = faiss_index_time
    
    return recommendations[['Car', 'Price', 'Mileage', 'Description', 'Cosine Similarity', 'FAISS Index Time']]

# Function to send a message to the model
def send_message(message, context=None):
    chat = model.start_chat(history=context)
    response = chat.send_message(message)
    return response

# Function to answer user queries based on retrieved data
def answer_with_retrieved_data(query, retrieved_data, backend_history):
    combined_message = f"User Query: {query}\n\nRetrieved Data:\n{retrieved_data}"
    context = backend_history + [{'role': 'user', 'parts': combined_message}]
    response = send_message(combined_message, context=context)
    assistant_text = response.to_dict()['candidates'][0]['content'].get('text', 'Details are not available right now.')
    return assistant_text

# Main function to handle queries and follow-up questions with response latency
def get_car_recommendations(message, visible_history, backend_history=None):
    backend_history = backend_history or [backend_instruction]  # Add the instruction at the start

    # Measure response start time
    response_start = time.time()
    
    response = send_message(message, context=backend_history)
    response_content = response.to_dict()['candidates'][0]['content']['parts'][0]
    function_call = response_content.get('function_call', {})
    function_args = function_call.get('args', {})

    # Extract price and preferences from user query
    semantic_search_query = function_args.get('semantic_search_query', "")
    max_price = function_args.get('max_price', 20000000)

    recommendations_text = ""
    if (semantic_search_query, max_price) != st.session_state.get('last_query', (None, None)):
        recommendations = get_required_cars(max_price=max_price, semantic_search_query=semantic_search_query)
        recommendations_text = "\n\n".join(
            f"**{row['Car']}**\n- Price: {row['Price']} INR\n- Mileage: {row['Mileage']} kmpl\n- Description: {row['Description']}\n- Cosine Similarity: {row['Cosine Similarity']:.4f}\n- FAISS Index Time: {row['FAISS Index Time']:.4f} seconds"
            for _, row in recommendations.iterrows()
        )
        st.session_state['last_recommendations'] = recommendations_text
        st.session_state['last_query'] = (semantic_search_query, max_price)

    # Measure response end time
    response_end = time.time()
    response_latency = response_end - response_start

    # Print response latency for debugging
    print(f"Response Latency: {response_latency:.4f} seconds")

    # Display recommendations with response latency
    if recommendations_text:
        combined_message = f"User Query: {message}\n\nRetrieved Data:\n{recommendations_text}\n\nResponse Latency: {response_latency:.4f} seconds"
        backend_history.append({'role': 'user', 'parts': combined_message})
        response = send_message(combined_message, context=backend_history)
        visible_history.append({'role': 'model', 'text': response.to_dict()['candidates'][0]['content']['parts'][0]['text']})
    else:
        retrieved_data = st.session_state.get('last_recommendations', '')
        assistant_text = answer_with_retrieved_data(message, retrieved_data, backend_history)
        visible_history.append({'role': 'model', 'text': assistant_text})

    # Print visible history for debugging
    print(visible_history)

    backend_history.append({'role': 'user', 'parts': message})
    print(backend_history)
    return response, visible_history, backend_history

# Streamlit Interface
st.title("Car Recommendation Chatbot")
st.write("A Recommendation Chatbot built to Tailor your car for you!")

# Initialize chat history in session state
if 'visible_history' not in st.session_state:
    st.session_state['visible_history'] = [{'role': 'model', 'text': 'I am here to recommend cars based on your preferences. Please provide me with your budget range in order to better understand your needs.'}]
if 'backend_history' not in st.session_state:
    st.session_state['backend_history'] = []

# Display conversation history
for entry in st.session_state['visible_history']:
    with st.chat_message(entry['role']):
        st.markdown(entry['text'])

# User input for chat
if user_input := st.chat_input("Your Query"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state['visible_history'].append({'role': 'user', 'text': user_input})

    response, st.session_state['visible_history'], st.session_state['backend_history'] = get_car_recommendations(
        user_input, st.session_state['visible_history'], st.session_state['backend_history']
    )

    # Display model response
    if response:
        with st.chat_message("model"):
            candidate_content = response.to_dict()['candidates'][0]['content']
            assistant_text = candidate_content.get('parts', [{}])[0].get('text', 'Details are not available right now.')
            st.markdown(assistant_text)
