
from typing import TypedDict, Sequence,Annotated
from langchain_core.messages import BaseMessage
import operator
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

class State(TypedDict):
    """This is is my graph state where store all conversations."""
    messages: Annotated[Sequence[BaseMessage], operator.add]  

class LlmAgent:
    """
    A class to represent an LLM agent that can process queries and provide responses.
    """

    def __init__(self, llm):
        self.llm =  llm
        
    def llm_call(self, state: State):
        try:
            print("--LLM Call--")
            question = state['messages'][0]
            prompt = PromptTemplate(
                template="""
                You are a helpful ai assistant. Answer the question based on you capability. if you think you do not know about the question do not give the answer just simply say i do not know the proper answer. Though do not give any hypothetical answer or hallucination response or do not predict the answer. If you are confident enough then provide the answer properly.\n
                the user query is: {question}
                """,
                input_variables=["question"]
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke(
                {
                    "question": question
                }
            )
            print(response)
            return {
                "messages": [response]
            }
        except Exception as e:
            print(f"Error in llm_call: {e}")
            return {
                "messages": [f"An error occurred: {e}"]
            }
    
  