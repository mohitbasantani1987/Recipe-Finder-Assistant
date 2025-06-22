import faiss
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal,TypedDict,Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain_huggingface import HuggingFaceEmbeddings
from IPython.display import display, Image
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from agents.llm import LlmAgent
from agents.rag import RagAgent
import streamlit as st
from agents.webSearchAgent import WebSearchAgent
import os
import json
load_dotenv()
from langchain_groq import ChatGroq

class State(TypedDict):
    """This is is my graph state where store all conversations."""
    messages: Annotated[Sequence[BaseMessage], operator.add]

class SelectPath(BaseModel):
    path: Annotated[Literal["LLM_CALL", "RAG_CALL", "WEB_SEARCH"], Field(description="Based on the user query select one path only. if user want to know about the Attention all you need paper or any query related to the Transformer then select the Rag call. if user want to know basics question the select llm call and finally user want to know about recent info then select the internet search.")]
    reason: Annotated[str, Field(description="Why you select a specific path give reason in one sentence.")]  

class validationOutput(BaseModel):
    validation_response: Annotated[Literal["yes", "no"], Field(description="analysis the user question and the output. If you think the output is correct based on the question then say 'yes' otherwise say 'no' if the answer is not correct based on the user question.")]      

class RecipeEngine:
    """
    A class to represent a recipe engine that processes recipes.
    """

    def __init__(self, recipe):
        self.recipe = recipe

    def loadConfig(self):
        os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
        os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
        os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")

    def process_split_chunk_vectorize_Pdf(self, pdf_path: str, embeddings: HuggingFaceEmbeddings) -> Dict[str, Any]:
        """
        Process a PDF file to extract recipe information.Also creates a vector store for the recipe data.
        """
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
                chunk_size = 500,
                chunk_overlap = 100,
                separators=["\n\n", "\n", " ", ""]
            )

        chunks = splitter.split_documents(documents=documents) 

        embeddings_length = len(embeddings.embed_query("Hello"))
        #print(embeddings_length)
        index = faiss.IndexFlatIP(embeddings_length)
        # create vector store
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
            )
        vector_store.add_documents(documents=chunks)
        
        return vector_store   
    
    ## Supervisor node
    def supervisor(self, state: State):
        """This node is responsible for to select the correct path based on the user query."""
        question = state['messages'][-1]
        print(question)
        parser = PydanticOutputParser(pydantic_object=SelectPath)
        path_prompt = PromptTemplate(
            template="""
            Think step by step on user query. The user query: {question}\n\n My vector store is related of Recipe Cookbook and the name is *recipe* so if you think like user query is related to this cookbook then select the 'RAG_CALL'. Or user want to know about the latest news or any info the you select the 'WEB_SEARCH'. Finally if user want to know basics question such as wants to know recipe of indian cusine,italian cusine, japenese cusine etc. or any basics question if you think llm model can capable to give sold and correct answer then select the 'LLM_CALL'. you have to follow this format instruction.\n\n{format_instruction}.
            """,
            input_variables=["question"],
            partial_variables={
                "format_instruction": parser.get_format_instructions()
            }
        )
        chain = path_prompt | llm | parser
        response = chain.invoke(question)
        print(response.path, response.reason)
        
        return {
            "messages": [response.path]
        }
    
    def router(self, state: State):
        print("--Router--")
        last_message = state["messages"][-1]
        print(last_message)
        
        if last_message == "RAG_CALL":
            return "RAG_CALL"
        elif last_message == "LLM_CALL":
            return "LLM_CALL"
        else:
            return "WEB_SEARCH"
        
    def validation_node(self, state: State):
        print("---validation---")
        question = state['messages'][0]
        answer = state['messages'][-1]
        validate_parser = StrOutputParser()
        print(question, answer)
        validation_prompt = PromptTemplate(
            template="""Think you are evaluator, your task is validate the output of the user question. Example if the answer is correct based on the user question then simple say 'yes', otherwise say 'no' is the answer is not relevant to the question. Think very carefully to validation time. Below the user question and the answer of the question.
            the user question is : {question}\n\n and the output answer of the question is: {answer}.\n\n Most important is must be follow the output format instruction: {format_instruction}""",
            input_variables=["question", "answer"],
            partial_variables={
                "format_instruction": PydanticOutputParser(pydantic_object=validationOutput).get_format_instructions()
            }
        )
        
        validation_chain = validation_prompt | llm | validate_parser
        
        raw_result = validation_chain.invoke(
            {
                "question": question,
                "answer": answer
            }
        )
        # Extract JSON from the LLM output
        match = re.search(r"\{.*\}", raw_result, re.DOTALL)
        if match:
            json_str = match.group(0)
            result = json.loads(json_str)
            print(result["validation_response"])
            return {
                "messages": [result["validation_response"]]
            }
        else:
            print("No valid JSON found in LLM output:", raw_result)
            return {
                "messages": ["no"]
            }
    
    def validation_router(self, state: State):
        validation_result = state['messages'][-1]
        
        if validation_result == "yes":
            return "PASS"
        else:
            return "FAILED"

    def final_answer(self, state: State):
        final_answer = state['messages'][2]
        print("final_result: ",final_answer)    
  

if __name__ == "__main__":
    recipe = {
        "ingredients": ["2 cups flour", "1 cup sugar", "1/2 cup butter"],
        "instructions": "Mix all ingredients and bake at 350F for 30 minutes."
    }
    
    engine = RecipeEngine(recipe)
    engine.loadConfig()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct") 
    
    # Process a PDF file (example path)
    pdf_path = "data/recipe.pdf"
    vector_store = engine.process_split_chunk_vectorize_Pdf(pdf_path, embeddings)
    
    result = vector_store.similarity_search(query="Get me the recipe of Mango Yogurt Drink", k=2)
    retriever = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs={
                "k": 3
            }
    )

    workflow = StateGraph(State)

    ## Node
    workflow.add_node("supervisor", engine.supervisor)
    workflow.add_node("LLM", LlmAgent(llm).llm_call)
    workflow.add_node("RAG", RagAgent(llm, retriever).rag_call)
    workflow.add_node("WEB", WebSearchAgent(llm).web_search)
    workflow.add_node("VALIDATION", engine.validation_node)
    workflow.add_node("END", engine.final_answer)

    ## Edges
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        engine.router,
        {
            "RAG_CALL": "RAG",
            "LLM_CALL": "LLM",
            "WEB_SEARCH": "WEB",
        }
    )
    workflow.add_edge("RAG", "VALIDATION")
    workflow.add_edge("LLM", "VALIDATION")
    workflow.add_edge("WEB", "VALIDATION")
    workflow.add_conditional_edges(
        "VALIDATION",
        engine.validation_router,
        {
            "PASS": END,
            "FAILED": "supervisor"
        }
    )

    app = workflow.compile()

    #display(Image(app.get_graph().draw_mermaid_png()))

    st.set_page_config(page_title="Recipe Finder Assistant!", layout="centered")
    st.title("🍲 Recipe Finder Assistant!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []    

    user_input = st.chat_input("Ask your today's recipe or any query related to the recipe cookbook...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        state = {"messages": [user_input]}
        result = app.invoke(state)
        
        st.session_state.messages.append({"role": "assistant", "content": result['messages'][-2]})     

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])    
