from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI


'''
Runnable are simply a method to define tasks. 
You just define a task and declare it as a runnable, and then you
can chain those tasks(runnables) in a sequence(chain) and then you execute the sequence.

Three ways to chain a runnable: 
1. First (single runnable)
2. Last (single runnable)
3. Middle (list of runnables)

RunnableSequence is a chain of runnables that execute a sequential manner like a -> b -> c - > d

RunnableParallel is a chain of runnables that execute in parallel manner.
'''

load_dotenv()

model = ChatOpenAI(model="gpt-4-mini")

messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)

# Create individual runnables (steps in the chain)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

response = chain.invoke({"topic": "lawyers", "joke_count": 3})

print(response)
