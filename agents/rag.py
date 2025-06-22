from typing import TypedDict, Sequence,Annotated
from langchain_core.messages import BaseMessage
import operator
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

class State(TypedDict):
    """This is is my graph state where store all conversations."""
    messages: Annotated[Sequence[BaseMessage], operator.add]  


class RagAgent:
    """
    A class to represent an RAG agent that can process queries and provide responses.
    """

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def format_docs(self, retriever_docs):
        context_text = "\n\n".join(doc.page_content for doc in retriever_docs)
        return context_text    

    def rag_call(self, state: State):
        try:
            print("--Rag call--")
            question = state['messages'][0]
            print(question)
            prompt = hub.pull("rlm/rag-prompt")
        
            rag_chain = (
                {
                    "context": self.retriever | self.format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
            response = rag_chain.invoke(question)
            print("rag-response", response)
            
            return {
                "messages": [response]
            }
        except Exception as e:
            print(f"An error occurred in rag_call: {e}")
            return {
                "messages": [f"Error: {e}"]
            }