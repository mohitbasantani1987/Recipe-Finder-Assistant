from typing import TypedDict, Sequence,Annotated
from langchain_core.messages import BaseMessage
import operator
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_tavily import TavilySearch

class State(TypedDict):
    """This is is my graph state where store all conversations."""
    messages: Annotated[Sequence[BaseMessage], operator.add]  


class WebSearchAgent:
    def __init__(self, llm):
        self.llm = llm

    
    def web_search(self, state: State):
        try:
            question = state['messages'][0]
            tavily_search = TavilySearch(
                max_results=3,
                topic="general"
            )
            web_search_content = tavily_search.invoke(question)
            prompt = PromptTemplate(
                template="""
                You are a helpful ai assistant who can give answer of the user query based on the web search content. First read the user query and check the web_search_content. If you think web content is not enough for to give the answer of the user query then simply say. No have enough context to give the answer. And below the user query and web_search_content.\n\n
                the user query is: {question}\n\n
                the web_search_content: {web_search_content}\n
                """,
                input_variables=["question", "web_search_content"]
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke(
                {
                    "question": question,
                    "web_search_content": web_search_content
                }
            )
            return {
                "messages": [response]
            }
        except Exception as e:
            return {
                "messages": [f"An error occurred during web search: {str(e)}"]
            }