import os
import openai

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
from langchain_core.tools import tool

@tool
def search(query:str)->str:
    return "42f"
from pydantic import BaseModel,Field
class SearchInput(BaseModel):
    query:str=Field(description="Thing to search for")
@tool
def search(query: str) -> str:
    """Search for the weather online."""
    return "42f"
print(search.args)
print(search.run("sf"))
import requests
from pydantic import BaseModel, Field
import datetime
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    # Parameters for the request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']

    # 从返回的逐小时数据里，找到离当前 UTC 时间最近的那个时间点对应的温度
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]

    return f'The current temperature is {current_temperature}°C'
print(get_current_temperature.name)
print(get_current_temperature.description)
print(get_current_temperature.args)
from langchain_community.tools.render import format_tool_to_openai_function
format_tool_to_openai_function(get_current_temperature)
get_current_temperature.invoke({"latitude": 13, "longitude": 14})

import wikipedia
@tool
def search_wikipedia(query: str) -> str:
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            wikipedia.exceptions.PageError,
            wikipedia.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)
print(search_wikipedia.name)
print(search_wikipedia.description)
print(format_tool_to_openai_function(search_wikipedia))
search_wikipedia.invoke({"query": "langchain"})
from langchain_classic.chains.openai_functions.openapi import openapi_spec_to_openai_fn
from langchain_community.utilities.openapi import OpenAPISpec
text = """
{
  "openapi": "3.0.0",
  "info": {
    "version": "1.0.0",
    "title": "Swagger Petstore",
    "license": {
      "name": "MIT"
    }
  },
  "servers": [
    {
      "url": "http://petstore.swagger.io/v1"
    }
  ],
  "paths": {
    "/pets": {
      "get": {
        "summary": "List all pets",
        "operationId": "listPets",
        "tags": [
          "pets"
        ],
        "parameters": [
          {
            "name": "limit",
            "in": "query",
            "description": "How many items to return at one time (max 100)",
            "required": false,
            "schema": {
              "type": "integer",
              "maximum": 100,
              "format": "int32"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "A paged array of pets",
            "headers": {
              "x-next": {
                "description": "A link to the next page of responses",
                "schema": {
                  "type": "string"
                }
              }
            },
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Pets"
                }
              }
            }
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      },
      "post": {
        "summary": "Create a pet",
        "operationId": "createPets",
        "tags": [
          "pets"
        ],
        "responses": {
          "201": {
            "description": "Null response"
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      }
    },
    "/pets/{petId}": {
      "get": {
        "summary": "Info for a specific pet",
        "operationId": "showPetById",
        "tags": [
          "pets"
        ],
        "parameters": [
          {
            "name": "petId",
            "in": "path",
            "required": true,
            "description": "The id of the pet to retrieve",
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Expected response to a valid request",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Pet"
                }
              }
            }
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "Pet": {
        "type": "object",
        "required": [
          "id",
          "name"
        ],
        "properties": {
          "id": {
            "type": "integer",
            "format": "int64"
          },
          "name": {
            "type": "string"
          },
          "tag": {
            "type": "string"
          }
        }
      },
      "Pets": {
        "type": "array",
        "maxItems": 100,
        "items": {
          "$ref": "#/components/schemas/Pet"
        }
      },
      "Error": {
        "type": "object",
        "required": [
          "code",
          "message"
        ],
        "properties": {
          "code": {
            "type": "integer",
            "format": "int32"
          },
          "message": {
            "type": "string"
          }
        }
      }
    }
  }
}
"""
spec = OpenAPISpec.from_text(text)
pet_openai_functions, pet_callables = openapi_spec_to_openai_fn(spec)
from langchain_openai import ChatOpenAI
model = ChatOpenAI(temperature=0).bind(functions=pet_openai_functions)
model.invoke("what are three pets names")
model.invoke("tell me about pet with id 42")
functions = [
    format_tool_to_openai_function(f) for f in [
        search_wikipedia, get_current_temperature
    ]
]
model = ChatOpenAI(temperature=0).bind(functions=functions)

model.invoke("what is the weather in sf right now")
model.invoke("what is langchain")
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    ("user", "{input}"),
])
chain = prompt | model
chain.invoke({"input": "what is the weather in sf right now"})
from langchain_classic.agents.output_parsers import OpenAIFunctionsAgentOutputParser
chain = prompt | model | OpenAIFunctionsAgentOutputParser()
result = chain.invoke({"input": "what is the weather in sf right now"})
print(result.tool_input)
get_current_temperature.invoke(result.tool_input)
result = chain.invoke({"input": "hi!"})
print(result.return_values)
from langchain_core.agents import AgentFinish
def route(result):
    if isinstance(result,AgentFinish):
        return result.return_values['output']
    else:
        tools={
            "search_wikipedia": search_wikipedia,
            "get_current_temperature": get_current_temperature,
        }
    return tools[result.tool].run(result.tool_input)


chain = prompt | model | OpenAIFunctionsAgentOutputParser() | route
result = chain.invoke({"input": "What is the weather in san francisco right now?"})
result = chain.invoke({"input": "What is langchain?"})
chain.invoke({"input": "hi!"})