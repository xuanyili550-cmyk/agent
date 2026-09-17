import os
from dotenv import load_dotenv
_=load_dotenv()

profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}
prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently."
}

email = {
    "from": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "body": """
Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",
}


from pydantic import BaseModel ,Field
from typing_extensions import TypedDict,Literal,Annotated
from langchain.chat_models import init_chat_model


llm=init_chat_model("openai:gpt-4o-mini")

class Router(BaseModel):
    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )
llm_router=llm.with_structured_output(Router)
from prompts import triage_system_prompt, triage_user_prompt
system_prompt=triage_system_prompt.format(
    full_name=profile["full_name"],
    name=profile["name"],
    examples=None,
    user_profile_background=profile["user_profile_background"],
    triage_no=prompt_instructions["triage_rules"]["ignore"],
    triage_notify=prompt_instructions["triage_rules"]["notify"],
    triage_email=prompt_instructions["triage_rules"]["respond"],
)

user_prompt = triage_user_prompt.format(
    author=email["from"],
    to=email["to"],
    subject=email["subject"],
    email_thread=email["body"],
)
result = llm_router.invoke(
    [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
)
print(result)
from langchain_core.tools import tool
@tool
def write_email(to:str,subject:str,content:str)->str:
    return f"Email sent to {to} with subject '{subject}'"

@tool
def schedule_meeting(
        attendees:list[str],
        subject:str,
        duration_minutes:int,
        preferred_day:str
)->str:
        return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"
@tool
def check_calendar_availability(day: str) -> str:
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"
from prompts import agent_system_prompt
def create_prompt(state):
    return [
        {
            "role":"system",
            "content":agent_system_prompt.format(
                instructions=prompt_instructions["agent_instructions"],
                **profile
            )
        }
    ]+ state['messages']
print(agent_system_prompt)
from langgraph.prebuilt import create_react_agent
tools=[write_email,schedule_meeting,check_calendar_availability]
agent=create_react_agent(
    "openai:gpt-4o",
    tools=tools,
    prompt=create_prompt,
)
response = agent.invoke(
    {"messages": [{
        "role": "user",
        "content": "what is my availability for tuesday?"
    }]}
)
print(response["messages"][-1].pretty_print())
from langgraph.graph import add_messages

class State(TypedDict):
    email_input:dict
    messages:Annotated[list,add_messages]

from langgraph.graph import StateGraph,START,END
from langgraph.types import Command
from typing import Literal
from IPython.display import Image,display
def triage_router(state:State)->Command[
    Literal["response_agent","__end__"]
]:
    author=state['email_input']['author']
    to=state['email_input']['to']
    subject=state['email_input']['subject']
    email_thread=state['email_input']['email_thread']
    system_prompt=triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
        examples=None
    )
    user_prompt=triage_user_prompt.format(
        author=author,
        to=to,
        subject=subject,
        email_thread=email_thread
    )
    result=llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    if result.classification == "respond":
        goto = "response_agent"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Respond to the email {state['email_input']}",
                }
            ]
        }
    elif result.classification == "ignore":
        update = None
        goto = END
    elif result.classification == "notify":
        update = None
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)

email_agent = StateGraph(State)
email_agent = email_agent.add_node(triage_router)
email_agent = email_agent.add_node("response_agent", agent)
email_agent = email_agent.add_edge(START, "triage_router")
email_agent = email_agent.compile()
display(Image(email_agent.get_graph(xray=True).draw_mermaid_png()))
email_input = {
    "author": "Marketing Team <marketing@amazingdeals.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "🔥 EXCLUSIVE OFFER: Limited Time Discount on Developer Tools! 🔥",
    "email_thread": """Dear Valued Developer,

Don't miss out on this INCREDIBLE opportunity!

🚀 For a LIMITED TIME ONLY, get 80% OFF on our Premium Developer Suite!

✨ FEATURES:
- Revolutionary AI-powered code completion
- Cloud-based development environment
- 24/7 customer support
- And much more!

💰 Regular Price: $999/month
🎉 YOUR SPECIAL PRICE: Just $199/month!

🕒 Hurry! This offer expires in:
24 HOURS ONLY!

Click here to claim your discount: https://amazingdeals.com/special-offer

Best regards,
Marketing Team
---
To unsubscribe, click here
""",
}
response = email_agent.invoke({"email_input": email_input})
email_input = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",
}
response = email_agent.invoke({"email_input": email_input})
for m in response["messages"]:
    m.pretty_print()



