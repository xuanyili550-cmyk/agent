import openai
import os

from dotenv import load_dotenv,find_dotenv
_=load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
from typing import List,Optional
from pydantic import BaseModel,Field
from langchain_community.utils.openai_functions import  convert_pydantic_to_openai_function

class Tagging(BaseModel):
    sentiment: str = Field(description="sentiment of text, should be `pos`, `neg`, or `neutral`")
    language: str = Field(description="language of text (should be ISO 639-1 code)")

convert_pydantic_to_openai_function(Tagging)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
model=ChatOpenAI(temperature=0)
tagging_functions=[convert_pydantic_to_openai_function(Tagging)]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Think carefully, and then tag the text as instructed"),
    ("user", "{input}")
])
model_with_functions = model.bind(
    functions=tagging_functions,
    function_call={"name": "Tagging"}
)
tagging_chain = prompt | model_with_functions
tagging_chain.invoke({"input": "I love langchain"})
tagging_chain.invoke({"input": "non mi piace questo cibo"})
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
tagging_chain = prompt | model_with_functions | JsonOutputFunctionsParser()
tagging_chain.invoke({"input": "non mi piace questo cibo"})
class Person(BaseModel):
    name: str = Field(description="person's name")
    age: Optional[int] = Field(default=None, description="person's age")
class Information(BaseModel):
    people: List[Person] = Field(description="List of info about people")

convert_pydantic_to_openai_function(Information)
extraction_functions = [convert_pydantic_to_openai_function(Information)]
extraction_model = model.bind(functions=extraction_functions, function_call={"name": "Information"})
extraction_model.invoke("Joe is 30, his mom is Martha")
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"),
    ("human", "{input}")
])
extraction_chain = prompt | extraction_model

extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})
extraction_chain = prompt | extraction_model | JsonOutputFunctionsParser()
extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})
from langchain_core.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="people")
extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})
os.environ.setdefault("USER_AGENT", "DeepLearningAI-LangChain-Course/1.0")
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
documents = loader.load()
doc = documents[0]
page_content = doc.page_content[:10000]
print(page_content[:1000])
class Overview(BaseModel):
    summary: str = Field(description="Provide a concise summary of the content.")
    language: str = Field(description="Provide the language that the content is written in.")
    keywords: str = Field(description="Provide keywords related to the content.")
overview_tagging_function = [
    convert_pydantic_to_openai_function(Overview)
]
tagging_model = model.bind(
    functions=overview_tagging_function,
    function_call={"name":"Overview"}
)
tagging_chain = prompt | tagging_model | JsonOutputFunctionsParser()
tagging_chain.invoke({"input": page_content})
class Paper(BaseModel):
    title: str
    author: Optional[str] = None
class Info(BaseModel):
    papers: List[Paper]
paper_extraction_function = [
    convert_pydantic_to_openai_function(Info)
]
extraction_model = model.bind(
    functions=paper_extraction_function,
    function_call={"name":"Info"}
)
extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")
extraction_chain.invoke({"input": page_content})
template = """A article will be passed to you. Extract from it all papers that are mentioned by this article follow by its author.

Do not extract the name of the article itself. If no papers are mentioned that's fine - you don't need to extract any! Just return an empty list.

Do not make up or guess ANY extra information. Only extract what exactly is in the text."""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{input}")
])
extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")

extraction_chain.invoke({"input": page_content})
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=0)
splits = text_splitter.split_text(doc.page_content)
def flatten(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list
flatten([[1, 2], [3, 4]])
from langchain_core.runnables import RunnableLambda
prep = RunnableLambda(
    lambda x: [{"input": doc} for doc in text_splitter.split_text(x)]
)
chain = prep | extraction_chain.map() | flatten
chain.invoke(doc.page_content)

