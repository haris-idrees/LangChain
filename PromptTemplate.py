from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

# Simplest approach to make Chat Prompts

# template = "Which country won {event}?"
#
# prompt_template = ChatPromptTemplate.from_template(template)
#
# context = {'event': "ICC champions Trophy 2017"}
#
# prompt = prompt_template.invoke(context)
#
# print(prompt)

# ______________________________________________________________________________________________

# Use multiple messages using tuples having roles and messages to make Prompts
messages = [
    ("system", "You are a helpful assistant."),
    ("human", "Did {country} won {event}?")
]

# Use specified messages to make Prompts

messages_2 = [
    ("system", "You are a helpful assistant."),
    HumanMessage("Did Pakistan won Icc Champions trophy 2017?"),
    AIMessage("No Pakistan lost Icc Champions Trophy 2017"),
    HumanMessage("No you are wrong try that again.")
]

# If you are specifically adding the role like the HumanMessage and AIMessage you won't be able to manipulate the
# context

messages_worng = [
    ("system", "You are a helpful assistant who guides {role}."),
    HumanMessage("Did {country} won {event}?"),
]  # This won't work as you have interpreted HumanMessage, to make it work you have to use tuple format
# like the messages = [("human", "Did {country} won {event}?")]

prompt_template = ChatPromptTemplate.from_messages(messages_2)
context = {'country': "Pakistan", 'event': "ICC champions Trophy 2017"}

prompt = prompt_template.invoke(context)

print(prompt)

result = model.invoke(prompt)

print(result.content)
