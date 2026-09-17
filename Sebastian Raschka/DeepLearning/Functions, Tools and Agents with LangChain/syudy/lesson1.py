import openai
import os
import json
from dotenv import load_dotenv,find_dotenv
_=load_dotenv(find_dotenv())
openai.api_key=os.environ["OPENAI_API_KEY"]
client=openai.OpenAI(api_key=openai.api_key)
def get_current_weather(location, unit="fahrenheit"):
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dump(weather_info)
functions=[{
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
}]
messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston?"
    }
]
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions
)

response_message = response.choices[0].message
json.loads(response_message.function_call.arguments)
args=json.loads(response_message.function_call.arguments)
get_current_weather(**args)
messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
)
messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call="auto",
)
print(response)
messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call="none",
)
print(response)
messages = [
    {
        "role": "user",
        "content": "What's the weather in Boston?",
    }
]
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call="none",
)
print(response)
messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call={"name": "get_current_weather"},
)
messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston!",
    }
]
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call={"name": "get_current_weather"},
)
print(response)
args = json.loads(response.choices[0].message.function_call.arguments)
observation = get_current_weather(**args)
messages.append(
        {
            "role": "function",
            "name": "get_current_weather",
            "content": observation,
        }
)