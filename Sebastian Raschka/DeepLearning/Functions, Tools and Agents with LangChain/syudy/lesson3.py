import openai
import os

from dotenv import load_dotenv,find_dotenv
_=load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
from typing import List
from pydantic import BaseModel,Field
class User:
    def __init__(self,name:str,age:int,email:str):
        self.name=name
        self.age=age
        self.email=email
foo = User(name="Joe",age=32, email="joe@gmail.com")
print(foo.name)
foo = User(name="Joe",age="bar", email="joe@gmail.com")
class pUser(BaseModel):
    name: str
    age: int
    email: str
foo_p = pUser(name="Jane", age=32, email="jane@gmail.com")
print(foo_p.name)
foo_p = pUser(name="Jane", age="bar", email="jane@gmail.com")
class Class(BaseModel):
    students: List[pUser]
obj=Class(
    students=[pUser(name="Jane", age=32, email="jane@gmail.com"),pUser(name="Jane1", age=32, email="jane@gmail.com")]
)
class WeatherSearch(BaseModel):
    airport_code:str=Field(description="airport code to get weather for")
from langchain_community.utils.openai_functions import convert_pydantic_to_openai_function
weather_function=convert_pydantic_to_openai_function(WeatherSearch)
class WeatherSearch1(BaseModel):
    airport_code: str = Field(description="airport code to get weather for")
convert_pydantic_to_openai_function(WeatherSearch1)
class WeatherSearch2(BaseModel):
    airport_code: str
convert_pydantic_to_openai_function(WeatherSearch2)
from langchain_openai import ChatOpenAI
model = ChatOpenAI()
model.invoke("what is the weather in SF today?", functions=[weather_function])
model_with_function = model.bind(functions=[weather_function])
model_with_function.invoke("what is the weather in sf?")
model_with_forced_function = model.bind(functions=[weather_function], function_call={"name":"WeatherSearch"})
model_with_forced_function.invoke("what is the weather in sf?")
model_with_forced_function.invoke("hi!")
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "{input}")
])
chain = prompt | model_with_function
chain.invoke({"input": "what is the weather in sf?"})
class ArtistSearch(BaseModel):
    artist_name: str = Field(description="name of artist to look up")
    n: int = Field(description="number of results")
functions=[
    convert_pydantic_to_openai_function(WeatherSearch),
    convert_pydantic_to_openai_function(ArtistSearch)
]
model_with_functions=model.bind(functions=functions)
model_with_functions.invoke("what is the weather in sf?")
model_with_functions.invoke("what are three songs by taylor swift?")
model_with_functions.invoke("hi!")

