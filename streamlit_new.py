import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai = OpenAI()

def generate_response_gpt(prompt, question=""):
    completion = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )
    return completion.choices[0].message.content

def check_response(input_question, output_response):
    prompt = f"""
        You are given an input question along with its response. Your task is to determine whether the response is relevant to the question.  

        ### **Instructions:**  
        - If the response **directly answers** the question and is **meaningful**, output **"valid"**.  
        - If the response is **off-topic, unclear, or does not address** the question, output **"irrelevant"**.  
        - If the user **intentionally wants to skip** the question, output **"skip"**.  

        ### **Input Details:**  
        **Question:** {input_question}  
        **Response:** {output_response}  

        ### **Expected Output:**  
        Provide only **one-word** output from the following:  
        ✅ **"valid"** – if the response appropriately answers the question.  
        ❌ **"irrelevant"** – if the response is unrelated or does not answer the question.  
        ➡️ **"skip"** – if the user explicitly chooses to skip the question.  
        """
    return generate_response_gpt(prompt, question="")

def recommend_chat(user_res, chat_history, list_questions):
    template = f"""
       # **Objective**  
        As an AI language model, your task is to generate a structured sequence of five relevant questions based on the provided conversation history. Each question should logically follow the previous one, ensuring coherence and continuity in the dialogue.  

        ---

        ## **Instructions**  

        ### **1. Review the Conversation History**  
        - Analyze the provided chat history to understand the context and flow of the conversation.  
        - Identify which questions have **already been asked and answered satisfactorily** to avoid repetition.  
        - Determine if the user's responses are **clear, complete, and relevant** to the preceding questions.  

        ### **2. Generate Relevant Questions**  
        - **If a question has already been asked and fully answered**, **do not ask it again**.  
        - **If a user's response is incomplete, unclear, or unsatisfactory**, rephrase and ask **the same question again** until a comprehensive answer is provided.  
        - If all previous responses are **satisfactory**, proceed with the next logical question in the sequence.  

        ### **3. Ensure Logical Flow & Coherence**  
        - Maintain a natural progression in the conversation by **building upon the user's previous answers**.  
        - The sequence should consist of **five unique and contextually relevant questions**.  
        - **Do not generate follow-up questions** beyond those listed below.  

        ---

        ## **Predefined List of Questions**  
        Only use the questions below, ensuring that none are repeated if already answered:  
       
        Generate Any question from following list of Questions (without changing the question's wording):

        Additional words could be used but question's wording would be as it is:

        {list_questions}

        ---

        Here is the previous Chat History

        ** Chat History **:  {chat_history}

        ## **Output Format**  
        - **Ask only one question at a time.**  
        - Ensure that each question is **not repeated** if already answered i.e. if present in chat_history.  
        - The next question should be **logically connected** to the previous answers.  
        """
    return generate_response_gpt(prompt=template, question=user_res)

def is_valid_response(question, response):
    """Checks if the user response is valid using check_response()."""
    result = check_response(question, response).lower()
    return "valid" in result or "skip" in result

def remove_asked_question(chat_history, questions):
    """Removes a question from the list if it was previously asked."""
    if not chat_history or len(chat_history) < 2:
        return

    last_ai_message = chat_history[-2].content.lower()

    for question in questions[:]:  # Create a copy to iterate over
        if all(word in last_ai_message for word in question.lower().split()):  
            questions.remove(question)
            break  # Stop at the first match

# Initialize session state
if 'questions' not in st.session_state:
    st.session_state.questions = [
        "What's motivating your job or service search right now?",
        "Would you describe your current work or mission as a calling?",
        "How would you articulate your life mission in a few words?",
        "How significant is your faith in influencing your career choices?",
        "Do you prefer working with organizations that share your faith or values?",
        "What is your current city and country of residence?",
        "What is your nationality?",
        "Are you open to relocating? If yes, where?",
        "What types of work are you interested in?",
        "What work environment do you prefer?"
    ]

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_question' not in st.session_state:
    st.session_state.current_question = None

if 'waiting_for_response' not in st.session_state:
    st.session_state.waiting_for_response = False

if 'conversation_active' not in st.session_state:
    st.session_state.conversation_active = True

# App layout
st.title("Career Conversation Assistant")

# Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)
    else:
        with st.chat_message("user"):
            st.write(msg.content)

# Conversation logic
if st.session_state.conversation_active:
    if not st.session_state.current_question and st.session_state.questions:
        # Start with first question
        st.session_state.current_question = st.session_state.questions.pop(0)
        st.session_state.waiting_for_response = True
        st.rerun()
    
    if st.session_state.waiting_for_response and st.session_state.current_question:
        # Display the current question if not already in chat history
        if not st.session_state.chat_history or st.session_state.chat_history[-1].content != st.session_state.current_question:
            with st.chat_message("assistant"):
                st.write(st.session_state.current_question)
        
        # Get user input
        if prompt := st.chat_input("Type your response here..."):
            if prompt.lower() == "quit":
                st.session_state.conversation_active = False
                with st.chat_message("assistant"):
                    st.success("Great, let me do a little science and show the Impacts that match your skills! 🚀")
                st.rerun()
            
            validation = check_response(st.session_state.current_question, prompt).lower()
            
            if "valid" in validation or "skip" in validation:
                # Add to chat history
                st.session_state.chat_history.extend([
                    AIMessage(content=st.session_state.current_question),
                    HumanMessage(content=prompt)
                ])
                
                # Remove asked question
                remove_asked_question(st.session_state.chat_history, st.session_state.questions)
                
                # Generate next question if available
                if st.session_state.questions and len(st.session_state.chat_history) < 10:
                    next_question = recommend_chat(
                        prompt,
                        st.session_state.chat_history,
                        st.session_state.questions
                    )
                    
                    # Update current question and remove from list if it matches
                    st.session_state.current_question = next_question
                    for q in st.session_state.questions[:]:
                        if q.lower() in next_question.lower():
                            st.session_state.questions.remove(q)
                            break
                    
                    st.session_state.waiting_for_response = True
                    st.rerun()
                else:
                    st.session_state.conversation_active = False
                    with st.chat_message("assistant"):
                        st.success("Conversation completed! Here are your results...")
            else:
                with st.chat_message("assistant"):
                    st.warning("Please provide a more specific answer to the question or type 'skip' to move on")
                st.session_state.waiting_for_response = True
                st.rerun()
else:
    with st.chat_message("assistant"):
        st.success("Thank you for completing the conversation!")