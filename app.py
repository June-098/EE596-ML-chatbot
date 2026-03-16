# Import the necessary libraries
import streamlit as st
from openai import OpenAI  
import numpy as np
import os
from dotenv import load_dotenv
from agent import Obnoxious_Agent,Head_Agent, Query_Agent, Answering_Agent, Relevant_Documents_Agent
from pinecone import Pinecone
import re

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

if "conv_hist" not in st.session_state:
    st.session_state["conv_hist"] = []

# Display existing chat messages
# ... (code for displaying messages)
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#Greetings:
GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|howdy|wassup|what'?s up|good\s+(morning|afternoon|evening)|how are you)\W*$",
    re.IGNORECASE
)

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
        if GREETING_PATTERNS.match(prompt):
            assistant_response = "Hello! How can I assist you with machine learning today?"
        else:
            obnoxious = st.session_state["head_agent"].Obnoxious_Agent.check_query(prompt)
            obnoxious_prompt = st.session_state["head_agent"].Obnoxious_Agent.extract_action(obnoxious)
            if obnoxious_prompt:
                assistant_response = "I can't answer this, because its obnoxious prompt"
            else:
                # Extract only the ML-relevant portion for hybrid queries
                # Include conv_hist so follow-up questions (e.g. "can you describe more?") resolve correctly
                extract_messages = [
                    {"role": "system", "content": "Extract only the machine learning or AI related question from the user's input. Machine learning topics include: decision trees, neural networks, perceptrons, gradient descent, SVMs, kernel methods, bias-variance tradeoff, unsupervised learning, clustering, classification, regression, ensemble methods, probabilistic modeling, gaussian distributions, Bayesian methods, covariate shift, domain adaptation, dimensionality reduction, kernel methods, expectation maximization, and any other statistics or math concepts commonly used in machine learning. If the user's message is a follow-up or refers to a previous topic (e.g. 'can you describe more?', 'give me an example'), rewrite it as a standalone ML question using the conversation history. If there is absolutely no ML-related question even considering prior context, return 'NONE'."},
                ]
                extract_messages.extend(st.session_state["conv_hist"])
                extract_messages.append({"role": "user", "content": prompt})
                extract_response = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=extract_messages
                )
                ml_query = extract_response.choices[0].message.content.strip()

                if ml_query.upper() == "NONE":
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
                        assistant_response = st.session_state["head_agent"].Answering_Agent.generate_response(ml_query, doc, st.session_state["conv_hist"])
                        st.session_state["conv_hist"].append({"role": "user", "content": ml_query})
                        st.session_state["conv_hist"].append({"role": "assistant", "content": assistant_response})
                    else:
                        assistant_response = "I couldn't find relevant documents to answer this question."
        st.markdown(assistant_response)

    # ... (append AI response to messages)
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
