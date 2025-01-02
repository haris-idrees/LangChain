from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama.llms import OllamaLLM


model_llama = OllamaLLM(model="llama3.2")

chat_history = [
    SystemMessage(content="Act as a helpful restaurant assistant.")
]


while True:

    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Assistant: Goodbye! Have a great day!")
        break

    chat_history.append(HumanMessage(content=user_input))

    response = model_llama.invoke(chat_history)

    chat_history.append(AIMessage(content=response))

    print(f"Assistant: {response}")


