import os
import tempfile
import streamlit as st
import asyncio
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from langchain import hub
from typing_extensions import List, TypedDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Initialize OpenAI embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)

# Initialize the language model
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Streamlit UI
st.title("Document-Based Q&A Chatbot")

# File uploader allows user to add their own PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Load and process the PDF document
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)

        # Add chunks to vector store
        vector_store.add_documents(documents=all_splits)

        # Define the prompt for question-answering
        prompt = hub.pull("rlm/rag-prompt")

        # Define the application state
        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str

        # Function to retrieve relevant documents
        def retrieve(state: State):
            retrieved_docs = vector_store.similarity_search(state["question"], k=5)
            return {"context": retrieved_docs}

        # Function to generate an answer based on the retrieved context
        def generate(state: State):
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = prompt.invoke({"question": state["question"], "context": docs_content})
            response = llm.invoke(messages)
            return {"answer": response.content}

        # Build the state graph for the application
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        # Initialize chat history and chat end state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "chat_ended" not in st.session_state:
            st.session_state.chat_ended = False

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Show chat input only if the chat has not ended
        if not st.session_state.chat_ended:
            user_input = st.chat_input("Enter your message here:")
            if user_input:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                # Check if the user wants to quit
                if user_input.strip().lower() == "quit":
                    st.session_state.chat_ended = True  # Mark chat as ended
                    st.rerun()  # Force rerun to remove chat input
                else:
                    # Generate assistant response using the knowledge graph
                    response = graph.invoke({"question": user_input})
                    assistant_response = response["answer"]

                    # Display assistant response
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    with st.chat_message("assistant"):
                        st.markdown(assistant_response)
        
        # If chat ended, show final message and remove input
        if st.session_state.chat_ended:
            st.info("Thanks for chatting with us.")

    finally:
        # Ensure the temporary file is deleted after processing
        os.remove(tmp_file_path)
