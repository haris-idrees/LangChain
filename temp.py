# from langchain_core.runnables import RunnableLambda, RunnableParallel
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain.schema.output_parser import StrOutputParser
# import os
#
# load_dotenv()
#
# API_KEY = os.getenv("OPENAI_API_KEY")
#
# if not API_KEY:
#     print("API Key not found.")
#     exit()
#
# model = ChatOpenAI(model="gpt-4o-mini")
# # def add_one(x: int) -> int:
# #     return x + 1
# #
# #
# # def mul_two(x: int) -> int:
# #     return x * 2
# #
# #
# # def mul_three(x: int) -> int:
# #     return x * 3
# #
# #
# # runnable_1 = RunnableLambda(add_one)
# # runnable_2 = RunnableLambda(mul_two)
# # runnable_3 = RunnableLambda(mul_three)
# #
# # sequence = runnable_1 | RunnableParallel(branches={"runable_2": runnable_2, "runnable_3": runnable_3})
# #
# # print(sequence.invoke(1))
#
#
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableParallel
# from langchain_openai import ChatOpenAI
#
#
# joke_chain = (
#     ChatPromptTemplate.from_template("tell me a joke about {topic}")
#     | model
# )
# poem_chain = (
#     ChatPromptTemplate.from_template("write a 2-line poem about {str}")
#     | model
# )
#
# runnable = RunnableParallel(joke=joke_chain, poem=poem_chain)
#
# context = {"topic": "programming", "str": "rain"}
#
# chain = runnable.invoke(context)
#
# print(chain)


from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI
import time

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

start_time = time.time()


def analyze_pros(x):
    return x * 3


def analyze_cons(x):
    return x * 2


def add(x, y):
    return x + y


pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x))
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x))
)

combine_results = (
    RunnableLambda(lambda x, y: add(x, y))
)

chain = (
    RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    # | RunnableLambda(lambda x: pros_branch_chain.invoke(x))
    # | RunnableLambda(lambda x: cons_branch_chain.invoke(x))
    | RunnableLambda(lambda x: add(x["branches"]["pros"], x["branches"]["cons"]))
)

result = chain.invoke(3)

end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")
print(result)
