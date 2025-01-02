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
print(current_dir)
file_path = os.path.join(current_dir, "docs", "attendance.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Creating Vector Store")

    if not os.path.exists(file_path):
        print("File does not exist. Please check the file path.")
        exit()

    loader = TextLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    print("Document chunks information")
    print("Number of chunks: ", len(docs))
    print("Sample Chunk: ", docs[0].page_content)

    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

    print("Finished creating embeddings")

    print("Creating Vector DB")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)

    print("Finished creating Vector DB")

    query = "what happens if a employee arrives after 15 minutes of "
else:
    print("Vector DB already exists")
