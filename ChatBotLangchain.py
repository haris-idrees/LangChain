from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
print(API_KEY)

if not API_KEY:
    print("API Key not found.")
    exit()

model = ChatOpenAI(model="gpt-4o-mini")

chat_history = [
    SystemMessage(content="Act as a helpful restaurant assistant. You have to guide the customer to place the order.")
]


while True:

    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Assistant: Goodbye! Have a great day!")
        break

    chat_history.append(HumanMessage(content=user_input))

    response = model.invoke(chat_history)

    chat_history.append(AIMessage(content=response.content))

    print(f"Assistant: {response.content}")


# LangChain also supports chat model inputs via strings or OpenAI format. The following are equivalent:
# model.invoke("Hello")
#
# model.invoke([{"role": "user", "content": "Hello"}])
#
# model.invoke([HumanMessage("Hello")])



# ________________________________________________________________________________________________________________
# Same chatbot as above but with a different prompt template

# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain.schema import AIMessage, HumanMessage, SystemMessage
#
# load_dotenv()
#
# model = ChatOpenAI(model="gpt-4o-mini")
#
# chat_history = []
#
# context = {
#     "name": "Bob",
#     "doctor": "Sage Pollack",
#     "first_name": "John",
#     "last_name": "Doe",
#     "member_id": "123456789",
#     "provider_id": "123456789",
#     "provider_name": "John Doe",
#     "provider_address": "123 Main St, Anytown, USA"
# }
#
# system_message = SystemMessage(content=
#                                f'''You are a call center AI bot. Your name is {context['name']}. You are calling on behalf of Dr
#                                 {context['doctor']}. Your goal is to gather insurance details of a patient with first name
#                                 {context['first_name']} and last name {context['last_name']}. The member ID of the patient is
#                                 {context['member_id']}.
#                                 User may also ask you some information about the provider like the Provider ID,
#                                 Provider Name, Provider Address. \n
#                                 User may provide some company information, and you must ignore it completely—no need to
#                                 reply to such information. User will also provide some options to select from menu
#                                 options, and you must select according to the following options. If some option is not
#                                 available, you must not reply to that.\n
#                                 Correct options are: \n
#                                 1. You are a provider, so if asked, you must select the provider option.\n
#                                 2. You are calling from a physical location, so select the related option.\n
#                                 3. You don’t know the party's extension, iso if asked say No or do not respond.\n
#                                 4. You want to talk to a representative, so select the related option.\n
#                                 Information to provide: \n
#                                 {context['provider_id']}\n
#                                 {context['provider_address']}\n
#                                 {context['provider_name']}\n
#                                 ''')
# chat_history.append(system_message)
#
# while True:
#     query = input("You: ")
#     if query.lower() == "exit":
#         break
#     chat_history.append(HumanMessage(content=query))
#
#     result = model.invoke(chat_history)
#     response = result.content
#     chat_history.append(AIMessage(content=response))
#
#     print(f"AI: {response}")
#
# print("---- Message History ----")
# print(chat_history)
