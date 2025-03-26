import openai
import os
from langchain.chat_models import init_chat_model

from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
load_dotenv()

file_path = "/home/nouman-aziz/Downloads/Grid Needs Assessment SCE 2024.pdf"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)


llm = init_chat_model("gpt-4o-mini", model_provider="openai")


async def load_pdf_async(file_path):
    """Asynchronously loads a PDF and returns its pages."""
    loader = PyPDFLoader(file_path)
    pages = [page async for page in loader.alazy_load()]
    docs = loader.load()
    return pages, docs


# Step 2: Execute the retrieval.
import asyncio
pages, docs = asyncio.run(load_pdf_async(file_path))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"],k=5)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
for i in range(10):
    user_input = input("Enter your question here: ")
    if user_input.lower() =="quit":
       break
    response = graph.invoke({"question": user_input})
    print(response["answer"])