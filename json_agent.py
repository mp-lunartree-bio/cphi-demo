from langchain_community.document_loaders import JSONLoader, CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI, AzureOpenAI

import json
from pathlib import Path
import pandas as pd
from typing import Callable, Dict, List, Optional, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

def get_or_create_db(file_name, embedding_function, recreate=False):
    file_path=f"./data/{file_name}.csv"

    # Define your persist directory
    persist_directory = f"./chroma_db_{file_name}"

    # Attempt to load the database
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    # Check if the database was loaded successfully
    df = pd.read_csv(file_path, encoding="utf-8")

    if not db or recreate:
        loader = CSVLoader(
            file_path=file_path,
            encoding="utf-8",
            csv_args={
                "delimiter": ",",
                "quotechar": '"',
                "fieldnames": df.columns.to_list(),
            },
        )
        documents = loader.load()

        # Create the database from documents if it wasn't loaded
        db = Chroma.from_documents(documents, embedding_function, persist_directory=persist_directory)
    
    return db

def get_retriever(db):
    retriever = db.as_retriever()
    return retriever

def get_similarity_search(query, db):
    docs = db.similarity_search(query, k=5)
    return docs

def get_prompt_template(user_messages, context_overview, context_people):
    template = f"""Answer the given question based on the following contexts:
    Context (overview of companies): {context_overview}
    Context (profile of people): {context_people}
    Context (past user messages): {[msg['content'] for msg in user_messages[:-1]]}

    The question below is asked by a user attending a pharma event. 
    Format of your answer could be different based on the context and the question.
    If the answer has a list of people or companies, please provide that as a bullet list.
    Make a coherent answer and provide a summary of relevance or importance of each company and person.

    Question: {user_messages[-1]['content']}
    """

    return template