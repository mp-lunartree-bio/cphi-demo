__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma

import pandas as pd

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

def get_prompt_template(messages, context_overview, context_people):
    past_messages = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[:-1]])
    
    template = f"""
    Answer the given question based on the following contexts:
    Context (overview of companies): {context_overview}
    Context (profile of people): {context_people}
    Context (past user messages): 
    {past_messages}

    The question below is asked by a user attending a pharma event. 
    Format your answer based on the context and the question.
    If the answer involves a list of people or companies, please provide it as a bullet list.
    Make a coherent answer and provide a summary of the relevance or importance of each company and person.

    Question: {messages[-1]['content']}
    """

    return template