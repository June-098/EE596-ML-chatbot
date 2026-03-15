# Import the necessary libraries
import streamlit as st
from openai import OpenAI  
import numpy as np
import os
from dotenv import load_dotenv
from agent import Obnoxious_Agent,Head_Agent, Query_Agent, Answering_Agent, Relevant_Documents_Agent
from pinecone import Pinecone

load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_key)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


st.title("Mini Project 2: Streamlit Chatbot")

# Define a function to get the conversation history (Not required for Part-2, will be useful in Part-3)
def get_conversation() -> str:
    #return: A formatted string representation of the conversation.
    #... (code for getting conversation history)
    if 'messages' in st.session_state:
        conversation = ""
        for message in st.session_state['messages']:
            conversation += f"{message['role']}: {message['content']}\n"
        return conversation
    else:
        return ""

# Check for existing session state variables
if "openai_model" not in st.session_state:
    # ... (initialize model)
    st.session_state["openai_model"] = "gpt-4.1-nano"


if "head_agent" not in st.session_state:
    st.session_state["head_agent"] = Head_Agent(openai_key, PINECONE_API_KEY, "machine-learning-textbook")

if "messages" not in st.session_state:
    # ... (initialize messages)
    st.session_state["messages"] = []

# Display existing chat messages
# ... (code for displaying messages)
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input

# Wait for user input
if prompt := st.chat_input("What would you like to chat about?"):
    # ... (append user message to messages)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    # ... (display user message)
    with st.chat_message("user"):
        st.markdown(prompt)
    # Generate AI response
    with st.chat_message("assistant"):
        obnoxious = st.session_state["head_agent"].Obnoxious_Agent.check_query(prompt)
        obnoxious_prompt = st.session_state["head_agent"].Obnoxious_Agent.extract_action(obnoxious)
        if obnoxious_prompt:
            assistant_response = "I can't answer this, because its obnoxious prompt"
        else:
            # Extract only the ML-relevant portion for hybrid queries
            extract_response = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": "system", "content": "Extract only the machine learning or AI related question from the user's input. If the entire input is ML-related, return it as-is. If there is no ML-related question, return 'NONE'."},
                    {"role": "user", "content": prompt}
                ]
            )
            ml_query = extract_response.choices[0].message.content.strip()

            if ml_query == "NONE":
                assistant_response = "I couln't answer this, because its irrelevant"
            else:
                doc = st.session_state["head_agent"].Query_Agent.query_vector_store(ml_query)
                print(f"Matches found: {len(doc.matches)}")
                if doc.matches:
                    print(f"First match metadat: {doc.matches[0].metadata.keys()}")
                    print(f"First match score: {doc.matches[0].score}")

                doc_text = [match.metadata.get('text', '') for match in doc.matches[:5]]
                context = "\n".join(doc_text)
                print(f"Text : {context[:500]}")
                if doc.matches and doc.matches[0].score > 0.4:
                    assistant_response = st.session_state["head_agent"].Answering_Agent.generate_response(ml_query, doc, st.session_state["messages"][:-1])
                else:
                    assistant_response = "I couldn't find relevant documents to answer this question."
        st.markdown(assistant_response)

    # ... (append AI response to messages)
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
