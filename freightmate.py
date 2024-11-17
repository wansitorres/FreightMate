import os
import openai
import numpy as np
import pandas as pd
import faiss
import streamlit as st
import warnings
from openai.embeddings_utils import get_embedding
from langchain_community.embeddings import OpenAIEmbeddings  # Updated import
from streamlit_option_menu import option_menu  # Ensure this import is present
from PIL import Image

warnings.filterwarnings("ignore")

# Streamlit app setup
st.set_page_config(page_title="FreightMate: Your Freight Selector AI Assistant", layout="wide")


System_Prompt = """
Role:
You are FreightMate, the "Freight Selector AI Assistant." Your role is to help businesses and logistics managers in the Philippines select the most appropriate freight mode (land, sea, or air) for specific shipments. You analyze shipment details, including the type of goods, weight, volume, urgency, distance, and budget to recommend the optimal freight mode that meets both logistical and cost requirements.

Instructions:
Analyze Shipment Data: Evaluate shipment details such as item type (perishable, fragile, bulky), weight, volume, urgency level, preferred delivery time, and budget.
Recommend Freight Mode: Based on the given shipment criteria, provide a freight mode recommendation (land, sea, or air). Consider the strengths and weaknesses of each mode relative to the shipment’s needs (speed, cost, capacity, handling).
Provide a Rationale: For each recommendation, explain why the suggested freight mode is the best choice based on the shipment details, referencing past data and logistics best practices. Be as detailed as possible when giving recommendations. Give an estimate of the cost and recommendations on where they can ship their freight with the freight mode recommended that is logical for the origin location.
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

# Sidebar for API key input
with st.sidebar:
    api_key = st.text_input('Enter OpenAI API token:', type='password')
    
    # Check if the API key is valid
    if api_key and api_key.startswith('sk-'):  # Removed length check
        openai.api_key = api_key
        st.success('API key is valid. Proceed to enter your shipment details!', icon='👉')
    else:
        st.warning('Please enter a valid OpenAI API token!', icon='⚠️')
    
    with st.container():
        l, m, r = st.columns((1, 3, 1))
        with l: st.empty()  # Corrected to call as a function
        with m: st.empty()  # Corrected to call as a function
        with r: st.empty()  # Corrected to call as a function
    
    options = option_menu(
        "Content",
        ["Home", "About Us", "FreightMate"]
    )

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

if options == "Home":
    st.markdown('<h1 style="color: #FFFFFF;">FreightMate: Your Freight Selector AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #FFFFFF;">Home</h2>', unsafe_allow_html=True)
    image = Image.open("E:/Downloads/freightmate.webp")
    resized_image = image.resize((300,300))
    st.image(resized_image)
    st.write("## Why Use Freightmate?")
    st.write("This tool is made to make your business decisions easier when it comes to Freight needs.") 
    st.write("With Freightmate, you can automate the process of figuring out the best Freight Mode for your deliveries!")
    st.write("Optimize your Freight decisions now!")
    st.write("## What it does")
    st.write("Input the weight, item type, urgency, budget, origin, and destination, and this tool will recommend the Freight Mode that best suits your needs, be it Land, Sea, or Air!")
    st.write("## Get Started")
    st.write("Click the FreightMate button on the sidebar to get started on Freight Mode recommendations!")

elif options == "About Us":
    st.markdown('<h1 style="color: #FFFFFF;">About Us</h1>', unsafe_allow_html=True)
    My_image = Image.open("E:/Downloads/About Me.jpg")
    my_resized_image = My_image.resize((300,300))
    st.image(my_resized_image)
    st.write("I am Juan Cesar E. Torres, a student in AI First Bootcamp by AI Republic.")
    st.write("This project is made as part of the requirements in the bootcamp.")
    st.write("I made this project to provide a solution for Logistics/Freight Problems in the Philippines.")
    st.write("Connect with me on LinkedIn or check out my other projects on Github!")
    st.write("https://www.linkedin.com/in/juan-cesar-torres-12b231260/")
    st.write("https://github.com/wansitorres")
    
elif options == "FreightMate":
    st.markdown('<h1 style="color: #FFFFFF;">FreightMate: Your Freight Selector AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #FFFFFF;">Input your details to get recommendations on what Freight Mode to use!</h1>', unsafe_allow_html=True)
    image = Image.open("E:/Downloads/freightmate.webp")
    resized_image = image.resize((300,300))
    st.image(resized_image)

    if api_key and api_key.startswith('sk-'):  # Removed length check
        weight = st.number_input("Enter the weight of the shipment (kg):", min_value=0)
        item_type = st.selectbox("Select the type of goods:", ["perishable", "fragile", "bulky", "other"])
        urgency = st.selectbox("Select urgency level:", ["1 day", "3 days", "7 days", "other"])
        budget = st.number_input("Enter your budget (PHP):", min_value=0)
        origin = st.text_input("Enter the origin location:")
        destination = st.text_input("Enter the destination location:")

        if st.button("Get Recommendation"):
        # Load the dataset and create embeddings only when the button is pressed
            dataframed = pd.read_csv('https://raw.githubusercontent.com/wansitorres/FreightMate/refs/heads/main/Freightdata.csv')
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