import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

openai = OpenAI()

def generate_response_gpt(prompt, question=""):
   
    # print("prompt is: ",prompt)
    completion = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": question,
            },
        ],
    )
    return completion.choices[0].message.content

def check_response(input,output):
    prompt = f"""
        You are given an input question along with its response. Your task is to determine whether the response is relevant to the question.  

        ### **Instructions:**  
        - If the response **directly answers** the question and is **meaningful**, output **"valid"**.  
        - If the response is **off-topic, unclear, or does not address** the question, output **"irrelevant"**.  
        - If the user **intentionally wants to skip** the question, output **"skip"**.  

        ### **Input Details:**  
        **Question:** {input}  
        **Response:** {output}  

        ### **Expected Output:**  
        Provide only **one-word** output from the following:  
        ✅ **"valid"** – if the response appropriately answers the question.  
        ❌ **"irrelevant"** – if the response is unrelated or does not answer the question.  
        ➡️ **"skip"** – if the user explicitly chooses to skip the question.  
        """

    final_res = generate_response_gpt(prompt, question="")

    return final_res


def recommend_chat(user_res, chat_history,list_questions):

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
        - Maintain a natural progression in the conversation by **building upon the user’s previous answers**.  
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

    res = generate_response_gpt(prompt=template, question=user_res)
    
    return res