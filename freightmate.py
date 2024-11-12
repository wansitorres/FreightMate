import os
import openai
import numpy as np
import pandas as pd
import faiss
import streamlit as st
import warnings

from openai.embeddings_utils import get_embedding
from langchain_community.embeddings import OpenAIEmbeddings  # Updated import

warnings.filterwarnings("ignore")

# Streamlit app setup
st.set_page_config(page_title="FreightMate: Your Freight Selector AI Assistant", layout="wide")

# Sidebar for API key input
with st.sidebar:
    api_key = st.text_input('Enter OpenAI API token:', type='password')
    
    # Check if the API key is valid
    if api_key and (api_key.startswith('sk-') and len(api_key) == 164):
        openai.api_key = api_key
        st.success('API key is valid. Proceed to enter your shipment details!', icon='👉')
    else:
        st.warning('Please enter a valid OpenAI API token!', icon='⚠️')

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# System prompt for the chatbot
System_Prompt = """
Role:
You are FreightMate, the "Freight Selector AI Assistant." Your role is to help businesses and logistics managers in the Philippines select the most appropriate freight mode (land, sea, or air) for specific shipments. You analyze shipment details, including the type of goods, weight, volume, urgency, distance, and budget to recommend the optimal freight mode that meets both logistical and cost requirements.

Instructions:
Analyze Shipment Data: Evaluate shipment details such as item type (perishable, fragile, bulky), weight, volume, urgency level, preferred delivery time, and budget.
Recommend Freight Mode: Based on the given shipment criteria, provide a freight mode recommendation (land, sea, or air). Consider the strengths and weaknesses of each mode relative to the shipment’s needs (speed, cost, capacity, handling).
Provide a Rationale: For each recommendation, explain why the suggested freight mode is the best choice based on the shipment details, referencing past data and logistics best practices. Be as detailed as possible when giving recommendations.
Stick to Freight Mode Recommendations: Limit responses strictly to the selection of freight modes. Do not answer questions outside this scope.
Answer Follow-Up Questions: You may also answer follow-up questions related to the shipment and make sure that you answer it properly.

Context:
FreightMate operates within the logistics industry in the Philippines, where different freight modes are chosen based on varying factors such as:

Air Freight: Best for high-urgency, perishable, or fragile goods that require fast delivery.
Sea Freight: Best for bulk shipments, low-urgency items, or heavy goods where cost efficiency is a priority.
Land Freight: Best for shipments within regions with shorter distances, moderate urgency, and a need for flexibility in delivery schedules.
FreightMate uses historical data and logistics expertise to make recommendations tailored specifically to the Philippine freight market.

Constraints:

Data Limitations: Your responses should only use the shipment details provided by the user. 
Scope of Assistance: Only respond to queries related to freight mode selection and anything related to it. 
Budget: Ensure that recommendations respect the user’s budget constraints, but be honest if the budget given is too low.
Time Sensitivity: Ensure that the recommended freight mode aligns with the required delivery timeframe.
Examples:

User Prompt:
"I have 300 kg of refrigerated seafood to ship from Davao to Manila. The delivery must be made within 1 day, and the budget is 20,000 PHP. Which freight mode would you recommend, and why?"

FreightMate Response:
"For this shipment, I recommend using Air Freight. Due to the high urgency (1-day delivery), refrigerated nature of the goods, and the need to maintain quality, air freight is the most suitable option. While air freight may be costlier, it offers the speed and temperature-controlled capabilities needed to meet both the urgency and perishability requirements within your budget."

User Prompt:
"We need to ship 5000 kg of rice from Nueva Ecija to Iloilo City. The shipment can take up to 7 days, and we are working with a budget of 8,000 PHP. What is the best freight mode?"

FreightMate Response:
"I recommend Sea Freight for this shipment. Since rice is a bulk, non-perishable item and the delivery time can be up to 7 days, sea freight offers the most cost-effective solution. With a shipment of this weight and your budget, sea freight will efficiently handle the load while keeping the costs low."

User Prompt:
"I need to ship 200 kg of electronic parts from Manila to Davao. Delivery must be completed within 3 days. My budget is 10,000 PHP. Which mode would be best?"

FreightMate Response:
"For this shipment, Air Freight is the best choice. The electronic parts are likely to be fragile and need quick delivery to meet the 3-day timeframe. Air freight ensures safe handling and the speed required, while staying within your budget constraints."
"""

# User input fields for shipment details
if api_key and (api_key.startswith('sk-') and len(api_key) == 164):
    st.title("FreightMate: Freight Selector AI Assistant")
    
    weight = st.number_input("Enter the weight of the shipment (kg):", min_value=0)
    item_type = st.selectbox("Select the type of goods:", ["perishable", "fragile", "bulky", "other"])
    urgency = st.selectbox("Select urgency level:", ["1 day", "3 days", "7 days", "other"])
    budget = st.number_input("Enter your budget (PHP):", min_value=0)
    origin = st.text_input("Enter the origin location:")
    destination = st.text_input("Enter the destination location:")

    if st.button("Get Recommendation"):
        # Load the dataset and create embeddings only when the button is pressed
        dataframed = pd.read_csv('https://raw.githubusercontent.com/wansitorres/freightmatedataset/refs/heads/main/Freightdata.csv')
        dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        documents = dataframed['combined'].tolist()

        # Generate embeddings for the documents
        embeddings = [get_embedding(doc, engine="text-embedding-3-small") for doc in documents]
        embedding_dim = len(embeddings[0])
        embeddings_np = np.array(embeddings).astype('float32')

        # Create a FAISS index for efficient similarity search
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings_np)

        user_message = f"I need to ship {weight} kg of {item_type} goods from {origin} to {destination}. The delivery needs to be completed in {urgency}, and the budget is {budget} PHP. Which freight mode should I choose, and why?"

        # Generate embedding for the user message
        query_embedding = get_embedding(user_message, engine='text-embedding-3-small')
        query_embedding_np = np.array([query_embedding]).astype('float32')

        # Search for similar documents
        _, indices = index.search(query_embedding_np, 2)
        retrieved_docs = [documents[i] for i in indices[0]]
        context = ' '.join(retrieved_docs)

        # Prepare structured prompt
        structured_prompt = f"Context:\n{context}\n\nQuery:\n{user_message}\n\nResponse:"

        # Call OpenAI API
        chat = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": System_Prompt}] + [{"role": "user", "content": structured_prompt}],
            temperature=0.5,
            max_tokens=1500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        # Get response
        response = chat.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display the response
        st.write(response)

    # Follow-up question input
    if st.session_state.messages:
        follow_up_question = st.text_input("Ask a follow-up question:")
        if st.button("Submit Follow-Up"):
            # Append the user's follow-up question to the messages
            st.session_state.messages.append({"role": "user", "content": follow_up_question})

            # Prepare the structured prompt for the follow-up question
            follow_up_context = ' '.join([msg['content'] for msg in st.session_state.messages if msg['role'] == 'assistant'])
            follow_up_prompt = f"Context:\n{follow_up_context}\n\nQuery:\n{follow_up_question}\n\nResponse:"

            # Call OpenAI API for the follow-up question
            follow_up_chat = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": System_Prompt}] + [{"role": "user", "content": follow_up_prompt}],
                temperature=0.5,
                max_tokens=1500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            # Get response for the follow-up question
            follow_up_response = follow_up_chat.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": follow_up_response})

            # Display the follow-up response
            st.write(follow_up_response)
else:
    st.warning("Please enter your OpenAI API key to use the chatbot.")