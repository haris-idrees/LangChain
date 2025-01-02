from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

import os

'''
This is a basic RAG implementation using LangChain and OpenAI
It takes a text file as input and splits it into chunks
It then uses OpenAI embeddings to create a vector database
'''

load_dotenv()

openai_api_ley = os.environ["OPENAI_API_KEY"]

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "docs", "attendance.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

if os.path.exists(persistent_directory):
    print("Persistent directory already exists")
    print("Deleting existing vector DB")
    os.system(f"rm -rf {persistent_directory}")

print("Creating new vector Store")

if not os.path.exists(file_path):
    print("File does not exist. Please check the file path.")
    exit()

loader = TextLoader(file_path)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

print("Document chunks information")
print("Number of chunks: ", len(docs))

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

print("Finished creating embeddings")

print("Creating Vector DB")
db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)

print("Finished creating Vector DB")

query = "what happens if a employee arrives 15 minutes after 10:00 AM?"


'''
The letter k defines the number of closest chunks related to the query. yani k wo chunks jin ko use kr k query to 
answer kia jaa sakta ha.
k=3 will return the top 3 closest relevant results.

The score threshold defines the minimum similarity score required for a chunk to be considered relevant. 
yani k chunks ka similarity score.
0.9 ka mtlb ha bht zyada similarity to os chunk ko relevant consider kren ge'''

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.9}
)

relevant_docs = retriever.invoke(query)

print("Relevant docs")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Doc {i}: {doc.page_content}")

