
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import streamlit as st
from dotenv import load_dotenv
import base64
import re
from PIL import Image
load_dotenv()
# from .get_info_db import get_info_by_name
from openai import OpenAI

class Model:
    def __init__(self):
        # os.environ["OPENAI_API_KEY"]= os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"]= st.secrets["OPENAI_API_KEY"]

        # os.getenv("OPENAI_API_KEY")
    
    def make_vectordb(self):
        pass
    
    def generate_response_gpt(self, prompt, question = ''):
        
        llm=ChatOpenAI(model="gpt-4",temperature=0.2)
        output_parser=StrOutputParser()
        chain=prompt|llm|output_parser
        res = chain.invoke({'info':f"{question}"})
        return res
    

    def generate_response_chat_completion(self, prompt, format=None):
        openai = OpenAI()

        # print("prompt is: ",prompt)
        if not format:
            completion = openai.chat.completions.create(
            model= "gpt-4o",
            messages= [
                { "role": "system", "content": prompt },

            
            ],
        )
        else:
            completion = openai.chat.completions.create(
            model= "gpt-4o",
            messages= [
                { "role": "system", "content": prompt },
            

            
            ],
            response_format={ "type": "json_object" }

        )
             
        return completion.choices[0].message.content
    def generate_response_chat_questions(self, prompt):
        openai = OpenAI()

        # print("prompt is: ",prompt)
        completion = openai.chat.completions.create(
        model= "gpt-4o",
        messages= [
            { "role": "system", "content": prompt },

        
        ],
    )
        return completion.choices[0].message.content
          
    def validate_answer(self, answer, next_question, questions, condition = False):
            
        prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", f"""
            You are a smart, empathetic conversational assistant designed to interact with experts. Your primary goal is to **verify their answers**, ensure **clarity and accuracy**, and **seamlessly guide the conversation forward**.

            ### Tone & Style
            - Be concise and conversational.
            - Warm, purpose-driven, supportive, conversational — but clear and efficient.
            - Use a light empathetic tone (e.g., "Understood", "Got it", "Makes sense").
            - Aim for minimal yet human acknowledgment (1–5 words).

            ### Conversation Flow
            - After validating the expert's answer, always ask the **next question** smoothly.
            - If the expert's answer is **irrelevant or unclear**, kindly ask for a **valid response** and **re-ask the same question**.
            - If the user replies with **"skip"**, respect that and move on without repeating the skipped question.

            ### Behavior Rules
            - Avoid repeating previously asked questions.
            - Maintain a natural, flowing transition between validation and the next question.
            - Your goal is to keep the expert engaged while collecting accurate responses.

            ### Context:
            - Already asked questions (to avoid repetition): {questions}
            - Next question to ask: {next_question}

           ### Input:
            User's answer: {answer}
                    """),
                    ("user", f"Answer: {answer}")
                ]
            )


            
        res = self.generate_response_gpt(prompt=prompt, question=next_question)
        return res
    
    def get_questions(self):
        # "experience", "expertise_analysis_in_disputes", "expertise_industries", "recognition",
        # "academic_position", "teaching", "previous_positions" , "testimony", "presentation",
        # "education","expertise_area_specialization", "services", "current_position", "services_ratings", "skills_ratings","management_level", "location", "availability", "open_to_relocation", "relocation_city" ,"register_as_firm", "skills"
        prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                """
                You are a conversational bot designed to interact directly with candidates in a human-like, engaging manner. Your goal is to gather detailed and comprehensive information from the candidate across multiple key areas, including both objective facts and subjective insights. This data will later be used for a matchmaking algorithm.

                    **Core Objectives:**
            - Gather detailed, truthful, and relevant information from the candidate.
            - Ask engaging, standalone questions—each must make sense without needing prior answers.
            - Adapt to context: Ask only what’s relevant based on information already provided.
            - Ensure the conversation is natural, respectful, and welcoming.

            **General Behavior Guidelines:**
            - Ask **1 relevant question per category** (minimum 5, maximum 8 total).
            - Prioritize **unanswered or partially complete** categories.
            - Avoid redundant or irrelevant questions.
            - Questions should **never assume a prior answer.** Each should be complete on its own.

            ---

            ### **Categories and Question Options**

            #### 1. Purpose & Calling
            1. What’s motivating your job or service search right now?  
            (Options: Live out faith, Meaningful work, New season, Values-aligned income, Other)
            2. Would you describe your current work/mission as a calling?  
            (Options: Yes deeply, Somewhat, Still exploring, Not really)
            3. In a few words, how would you describe your calling or life mission?  
            (Short text)

            #### 2. Faith Alignment
            4. How important is your faith in shaping your work decisions?  
            (Options: Central, Important, Open to it, Prefer not to say)
            5. Do you prefer working with organizations that share your faith or values?  
            (Options: Yes, It's a plus, No preference)

            #### 3. Location & Work Modality
            6. What is your current city and country of residence?
            7. What is your nationality?
            8. Are you open to relocating? (Options: Yes, Maybe, No)
            9. If yes or maybe, where would you consider relocating to? (Short text or multi-select)
            10. What types of work are you open to?  
                (Options: Full-time, Part-time, Contract, Volunteer, Project-based, Internship)
            11. What work environment do you prefer?  
                (Options: Remote, In-person, Hybrid, No preference)

            #### 4. Language Proficiency
            12. Which languages can you work in professionally?  
                (For each, select: Fluent / Business proficient)

            #### 5. Visa & Relocation Support
            13. Do you currently have a work visa for another country?  
                (Options: Yes, Applying, Need support, Not planning to)
            14. Which countries are you open to working in or moving to? (Text or multi-select)
            15. Are you open to receiving help with visas and relocation? (Yes/No)

            #### 6. Compensation
            16. What is your minimum monthly income requirement (in USD)?
            17. Are you open to roles that offer housing or other support instead of salary?  
                (Options: Yes, Maybe, No)

            #### 7. Experience, Skills & Strengths
            18. How many years of professional/service experience do you have?  
                (Options: <1, 1–3, 3–5, 5–10, 10+)
            19. What best describes your past roles/areas of service? (Select up to 5)
                (e.g., Admin, Ministry, Teaching, Tech, Healthcare, etc.)
            20. What are your top 3 work-related strengths?  
                (e.g., Leadership, Empathy, Creativity – multi-select with Other)
            21. What type of culture helps you thrive?  
                (e.g., Structured, Relational, Fast-paced, Mission-first)
            22. Are you comfortable learning new skills on the job?  
                (Options: Very, Somewhat, Prefer not to)

            #### 8. Availability & Final Snapshot
            23. When are you available to start?  
                (Options: Immediately, Within 30 days, 1–3 months, Flexible)
            24. Do you have references we could contact later?  
                (Options: Yes, Not yet, Prefer not to say)
            25. Upload your resume or CV (optional).
            26. Would you like VoGo to format your resume to our global standard?  
                (Yes, No)
            27. Anything else you’d like to add about your story or background?  
                (Short text)
            28. Would you like to connect with a professional job coach for personalized support?  
                (Yes/No)

            ---

            **Final Bot Behavior Recap:**

            - Maintain a warm, conversational, and efficient tone throughout.

            - Ask only one clear question per message—no multi-part or compound questions.

            - Do not request sensitive personal details such as phone numbers or home addresses.

            - Ensure every question is easy to understand and free of jargon.

            - Generate 8 different question each from 1 category stated above.

            - In first question add the introduction of the chatbot as well here is the introduction of the chatbot. 
               
               "Hi there! I’m your VoGo assistant. Let’s take 3–4 minutes to understand your purpose, experience, and preferences so we can connect you with meaningful opportunities."
               
               Introduction should be changed a bit not used as it is..

            - Avoid using section titles or headings like "Location & Work Modality" or "Purpose & Calling" etc. in your response.

            - Ask direct, concise questions without explaining the reason behind them or what will be done with the information.
            For example: Instead of saying, "What is your minimum monthly income requirement in USD? This will help us match you with opportunities that meet your financial needs," simply ask, "What is your minimum monthly income requirement in USD?"

            - When listing examples or choices, weave them naturally into the question.
            For example: “Which type of role are you open to —full-time, part-time, contract, volunteer, project-based, or internship?”

                    """
                            )
                        ]
                    )
        # info = get_info_by_name(email)
        # question = f"""{info}"""
        gt = self.generate_response_gpt(prompt)
        return gt


    def convert_csv_to_text(self, path):
        pass

    def get_first_response(self, question,Answer, context):
            prompt=ChatPromptTemplate.from_messages(
            [
                ("system",f""" You are a conversational bot for gathering information from Experts, your core purpose is to validate the answer from question. if answer is in context or indirectly related reply "yes" else manage accordingly. 
                 and provide assistance in case user's answer is irrelevant or user want to stop the process, 
                 there will be a context for you that contains previous questions and answers to the user.
                 
                 **Tone:** 
                - Warm, purpose-driven, supportive, conversational — but clear and efficient.
                 These are the fields : 
                 ### **Categories and Question Options**

            1. Purpose & Calling, 2. Faith Alignment, 3. Location & Work Modality,4. Language Proficiency,5. Visa & Relocation Support,
            6. Compensation,
            7. Experience, Skills & Strengths,
            8. Availability & Final Snapshot

                 If user says it does not have information or have not any experience, consider it in context.
                 
                 If required information is not provided, consider it out of context.
                 
                 if response is valid or other then ask the next question. else response is irrelevant ask the same question again.
                If the user wants to skip the question, responding with the word "skip" should be accepted.
             
                 If the user wants to skip the question, responding with the word "skip" should be accepted.
                                  
                 
                 STRICTLY FOLLOW THIS IF ANSWER IS IN CONTEXT: Most of the questions are from these fields. If the answer is in context Your response MUST be like: yes,Field Name(according to above mentioned): Answer
                 
                 if expert disowns (e.g. I have no such experience or skills (or any other field in above mentioned fields)) his current field value it should be still considered as valid response with yes, included in it.

                 IF answer is within context. MUST be always like: yes,Field Name(according to above mentioned): Answer
                 
                 context: {context}
                 Question: {question}
                 """),
                ("user",f"Answer:{Answer}")
            ]
                )
        
            gt = self.generate_response_gpt(prompt=prompt, question=question)

            return gt
        
        
    def verify_failed_response(self, question,Answer, response):
        prompt=ChatPromptTemplate.from_messages(
        [
            ("system",f""" You are a conversational bot for gathering information from Experts, your core purpose is to validate the answer from question.
                
                
                These are the fields : 
                1. Purpose & Calling, 2. Faith Alignment, 3. Location & Work Modality,4. Language Proficiency,5. Visa & Relocation Support,
                6. Compensation,
                7. Experience, Skills & Strengths,
                8. Availability & Final Snapshot
    
             Answer should be generated as 1st person e.g. "I have experience in Machine Learning"
               VERIFY THIS PATTERN: yes,Field Name(according to above mentioned): Answer
               
               If it's the same pattern response with exact same answer otherwise convert it into this pattern according to above fields names and Questions and Answer provided.

                DO NOT ANSWER ANYTHING ELSE BESIDE THAT.
                Question: {question}
                Answer: {Answer}
                
                
                """),
            ("user",f"Answer:{Answer}")
        ]
            )
    
        gt = self.generate_response_gpt(prompt=prompt, question=question)

        return gt
    
    def verify_irrelevant(self, response):
        prompt=ChatPromptTemplate.from_messages(
        [
            ("system",f""" You are a conversational bot for gathering to validate the text if it says that answer is not  relevant or not in the context or there is some confusion.

            VERIFY response if its irrelevant or any other problem/confusion it seems or it is still a question respond: "irrelevant"
            If it does not say anything like its not relevant or answer is within context respond: "valid"
            Intelligently understand that If User want to skip the question or want to move to next question in response then respond: "other"
            For confirmation-type questions, a simple "yes" should be considered a valid response.
            If the user wants to skip the question, responding with the word "skip" should be accepted.
            
             In short.
            If the response is unrelated to the current question, reply with "irrelevant." Similarly, for ambiguous responses such as "I don't know" or "I have no such information," treat them as "irrelevant."
             If user want to skip question then response with "other".
             and if user respond with proper answer then respond with "valid".
              
            If required information is not provided, consider it as irrelevant.
            If it's the same pattern response with exact same answer otherwise convert it into this pattern according to above fields names.
                
            Response: {response}
            
                """),
            ("user",f"Response:{response}")
        ]
            )
    
        gt = self.generate_response_gpt(prompt=prompt, question=response)
        return gt

def extract_questions(data):
    """
    Extracts and returns only the question text, removing any field numbers and labels before a colon.
    """
    return [item.split(": ", 1)[1] if ": " in item else item for item in data]


def continue_chat(session_state, session_key, message):
    questions = session_state['questions']
    current_question_index = session_state['current_question_index']
    cont = session_state['cont']

    if current_question_index >= len(questions):
        return {
            'message': 'You’re all set! We’ll match you with Growers aligned with your strengths,mission, and calling.',
            'completed': True
        }

    context = ""
    for index, (key, value) in enumerate(cont.items(), start=1):
        context += f"Question No {index}: {key} and Answer was: {value} \n"

    question = questions[current_question_index]

    response = model.get_first_response(
        question=question, 
        Answer=message, 
        context=context
    )

    cont[question] = response
    if len(cont) > 8:
        oldest_question = list(cont.keys())[0]
        del cont[oldest_question]

    session_state['cont'] = cont

    irrelevant = model.verify_irrelevant(response=response)
    print("irrelevant is:", irrelevant)

    if "other" == irrelevant.lower():
        current_question_index += 1
        if len(questions) <= current_question_index:
            return {
                'message': "You’re all set! We’ll match you with Growers aligned with your strengths,mission, and calling.",
                'questions_remaining': len(questions) - current_question_index,
                'completed': True
            }

        next_question = model.validate_answer(
            answer=response, 
            next_question=questions[current_question_index], 
            questions=questions[:current_question_index]
        )
        questions[current_question_index] = next_question

        session_state.update({
            'questions': questions,
            'current_question_index': current_question_index,
            'cont': cont
        })

        return {
            'message': questions[current_question_index],
            'irrelevant': True,
            'questions_remaining': len(questions) - current_question_index
        }

    elif "irrelevant" in irrelevant.lower() and "yes" not in response.split(',')[0].strip().lower():
        print("current question is: ", questions[current_question_index])
        next_question = model.validate_answer(
            answer=response, 
            next_question=questions[current_question_index], 
            questions=questions[:current_question_index]
        )
        
        print("Next question is: ", next_question)

        questions[current_question_index] = next_question
        session_state.update({
            'questions': questions,
            'current_question_index': current_question_index,
            'cont': cont
        })
        return {
            'message': next_question,
            'irrelevant': True
        }
    else:
        current_question_index += 1
        if current_question_index < len(questions):
            next_question = model.validate_answer(
                answer=response, 
                next_question=questions[current_question_index], 
                questions=questions[:current_question_index]
            )
            questions[current_question_index] = next_question

        session_state.update({
            'questions': questions,
            'current_question_index': current_question_index,
            'cont': cont
        })

        if current_question_index >= len(questions):
            next_step = "You’re all set! We’ll match you with Growers aligned with your strengths,mission, and calling."
        else:
            next_step = questions[current_question_index]

        return {
            'message': next_step,
            'questions_remaining': len(questions) - current_question_index,
            'completed': current_question_index >= len(questions)
        }

# Initialize the model
model = Model()

# # Set up session state
# if 'session_state' not in st.session_state:
#     # Get questions from the model
#     questions = model.get_questions()
#     data = questions.split("\n")
#     data = [item for item in data if item.strip() != ""]
#     questions_list = extract_questions(data)
    
#     print("questions_list length is: ", len(questions_list))
#     print("questions_list is: ", questions_list)

    
#     st.session_state.session_state = {
#         'questions': questions_list,
#         'current_question_index': 0,
#         'cont': {}
#     }
#     st.session_state.messages = []
#     st.session_state.session_key = "user_session"
    
#     # Add the first question immediately
#     first_question = questions_list[0]
#     st.session_state.messages.append({"role": "assistant", "content": first_question})

# # Set up the app layout
# st.title("Expert Information Gathering Chatbot")
# st.write("Please answer the following questions to help us understand your expertise.")

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
    

# # Get user input
# if prompt := st.chat_input("Your answer"):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # Display user message
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     # Get chatbot response
#     response = continue_chat(
#         st.session_state.session_state,
#         st.session_state.session_key,
#         prompt
#     )
    
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response['message']})
    
#     # Display assistant response
#     with st.chat_message("assistant"):
#         st.markdown(response['message'])
    
#     # Update session state
#     st.session_state.session_state = st.session_state.session_state
    
#     # Check if conversation is complete
#     if response.get('completed', False):
#         st.success("Thank you for completing all the questions!")
#         st.balloons()

# Set up session state


# if 'session_state' not in st.session_state:
#     # Get questions from the model
#     questions = model.get_questions()
#     data = questions.split("\n")
#     data = [item for item in data if item.strip() != ""]
#     questions_list = extract_questions(data)
    
#     st.session_state.session_state = {
#         'questions': questions_list,
#         'current_question_index': 0,
#         'cont': {}
#     }
#     st.session_state.session_key = "user_session"
#     st.session_state.current_question = questions_list[0]

# # Set up the app layout
# st.title("Expert Information Gathering Chatbot")
# st.write("Please answer the following questions to help us understand your expertise.")

# # Display current question
# with st.chat_message("assistant"):
#     st.markdown(st.session_state.current_question)

# # Get user input
# if prompt := st.chat_input("Your answer"):
#     # Clear previous content
#     st.empty()
    
#     # Get chatbot response
#     response = continue_chat(
#         st.session_state.session_state,
#         st.session_state.session_key,
#         prompt
#     )
    
#     # Update session state with new question
#     st.session_state.session_state = st.session_state.session_state
#     st.session_state.current_question = response['message']
    
#     # Rerun to show new question
#     st.rerun()
    
#     # Check if conversation is complete
#     if response.get('completed', False):
#         st.success("Thank you for completing all the questions!")
#         st.balloons()


# Function to convert image to base64 with resizing
def img_to_base64(image_path, size=(100, 100)):
    try:
        img = Image.open(image_path)
        img = img.resize(size)  # Resize the image
        img.save("temp_resized.png")  # Save resized version temporarily
        with open("temp_resized.png", "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        os.remove("temp_resized.png")  # Clean up temporary file
        return encoded
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return ""

# Load your company logo (replace with your actual image path)
logo_path = "V_favicon_Black.png"  # Update this path
logo_base64 = img_to_base64(logo_path, size=(150, 150))  # Increased avatar size

# Custom CSS for larger font and avatar styling
st.markdown(
    f"""
    <style>
        /* Larger font for all chat messages */
        .stChatMessage {{
            font-size: 24px !important;
        }}
        
        /* Avatar styling */
        .stChatMessage img[alt="company-logo"] {{
            border-radius: 70%;
            object-fit: cover;
            width: 50px !important;
            height: 50px !important;
            min-width: 50px !important;
            min-height: 50px !important;
        }}
        
        /* Input box font size */
        .stTextInput input {{
            font-size: 24px !important;
            padding: 20px !important;
        }}
        
        /* Success message font size */
        .stSuccess {{
            font-size: 24px !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Set up session state
if 'session_state' not in st.session_state:
    # Get questions from the model
    questions = model.get_questions()
    questions = questions.replace("**", "").replace("---", "")
    print("questions are: ", questions)
    data = questions.split("\n")
    data = [item for item in data if item.strip() != ""]
    questions_list = extract_questions(data)
    phrases_to_remove = [
        'Purpose & Calling',
        'Faith Alignment',
        'Location & Work Modality',
        'Language Proficiency',
        'Visa & Relocation Support',
        'Compensation',
        'Experience, Skills & Strengths',
        'Availability & Final Snapshot'
    ]
    questions_list = [question.replace('**', '').replace("---","") for question in questions_list]
    # Filter the list
    questions_list = [item for item in questions_list 
                     if not any(item.startswith(phrase) for phrase in phrases_to_remove)]
    
    questions_list = [item for item in questions_list if item.strip() != ""]

    questions_list = [re.sub(r'^\d+\.\s*', '', q) for q in questions_list]


    print("questions_list length is: ", len(questions_list))    
    print("questions_list is: ", questions_list)
    
    st.session_state.session_state = {
        'questions': questions_list,
        'current_question_index': 0,
        'cont': {}
    }
    st.session_state.session_key = "user_session"
    st.session_state.current_question = questions_list[0]

# Set up the app layout with larger title font
st.markdown("<h1 style='font-size: 32px;'>Expert Information Gathering Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<div style='font-size: 24px;'>Please answer the following questions to help us understand your expertise.</div>", unsafe_allow_html=True)

# Display current question with custom avatar
with st.chat_message("assistant", avatar=f"data:image/png;base64,{logo_base64}"):
    st.markdown(f"<div style='font-size:24px'>{st.session_state.current_question}</div>", unsafe_allow_html=True)

# Get user input with larger font
if not st.session_state.session_state.get('completed', False):
    if prompt := st.chat_input("Your answer"):
        # Clear previous content
        st.empty()
        
        # Get chatbot response
        response = continue_chat(
            st.session_state.session_state,
            st.session_state.session_key,
            prompt
        )
        
        # Update session state with new question
        st.session_state.session_state = response.get('session_state', st.session_state.session_state)
        
        # Check if conversation is complete
        if response.get('completed', False):
            st.session_state.session_state['completed'] = True  # Mark as completed
            # Clear any previous messages
            st.empty()
            # Display final success message with balloons
            with st.chat_message("assistant", avatar=f"data:image/png;base64,{logo_base64}"):
                st.markdown(
                    f"<div style='color:green; font-size:24px'>{response['message']}</div>", 
                    unsafe_allow_html=True
                )
                st.balloons()
            # Conversation is completed - show only the final message
            
        else:
            # Update current question only if not completed
            st.session_state.current_question = response['message']
            st.rerun()  # Show next question
else:
    # Conversation is completed - show only the final message
    st.empty()
    with st.chat_message("assistant", avatar=f"data:image/png;base64,{logo_base64}"):
        st.markdown(
            f"<div style='color:green; font-size:24px'>{st.session_state.session_state.get('final_message', 'Conversation Already completed!')}</div>", 
            unsafe_allow_html=True
        )