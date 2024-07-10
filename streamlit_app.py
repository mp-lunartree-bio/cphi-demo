import streamlit as st
from openai import OpenAI, AzureOpenAI
import pandas as pd
import json
from langchain_huggingface import HuggingFaceEmbeddings
from json_agent import get_or_create_db, get_retriever, get_similarity_search, get_prompt_template

openai_api_key = st.secrets["AZURE_OPENAI_API_KEY"]
openai_ep = st.secrets["AZURE_OPENAI_ENDPOINT"]
openai_ver = "2024-02-01"

if not openai_api_key:
    raise ValueError("Please enter your OpenAI API key.")
else:
    client = AzureOpenAI(
        azure_endpoint=openai_ep,
        api_key=openai_api_key,
        api_version=openai_ver,
    )

df_ex = pd.read_csv("./data/exhibitors.csv")

print("Loadig embedding function")
@st.cache_data
def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embedding_function = get_embedding_function()
print("Loaded embedding function")

print("Loading embeddings dbs")
db_overview = get_or_create_db('overview', embedding_function, recreate=False)
db_people = get_or_create_db('people', embedding_function, recreate=False)
print("Loaded embeddings dbs")

retriever_overview = get_retriever(db_overview)
retriever_people = get_retriever(db_people)

st.session_state.types = df_ex['Company Segment'].unique()
st.session_state.companies = df_ex.to_dict(orient="records")

st.session_state.relevant_types = []
st.session_state.relevant_companies = []
st.session_state.relevant_individuals = []

def format_data(data):
    print(data)
    if isinstance(data, list):
        return ", ".join([format_data(item) for item in data])
    elif isinstance(data, dict):
        return ", ".join([f"{key}: {str(value)}" for key, value in data.items()])
    elif isinstance(data, str):
        return data
    else:
        return str(data)

def format_prompt(role, description, data, example):
    return f"""
    You are an assistant that helps find the right {role} to meet at CPHI.
    Below are the {description} at the event: {format_data(data)}
    Your job is to give a JSON list of these {role}s based on the information provided.
    Your output should be a valid JSON of the names from the list with the key "list". 
    Example: {{"list": {example}}}
    """

def format_response_prompt(role, data):
    return f"""
    You are an assistant that tells why the given {role} are relevant to meet at CPHI based on the information provided by the user.
    Give the list of {role} in bullet points with details of the company information.
    Below are the names of relevant {role} at the event: {format_data(data)}.
    """

type_system_prompt = lambda data: format_prompt("category of companies", "types of exhibitor companies", data, '["Pharmaceutical Manufacturing", "Pharmaceutical Machinery"]')
company_system_prompt = lambda data: format_prompt("companies", "names of exhibitor companies", data, '["ACG Capsules", "Pfizer"]')
person_system_prompt = lambda data: format_prompt("person", "names of individuals", data, '["John Doe", "Sam Altman"]')

type_response_prompt = lambda data: format_response_prompt("categories of companies", data)
company_response_prompt = lambda data: format_response_prompt("companies", data)
person_response_prompt = lambda data: format_response_prompt("individuals", data)

def get_json_response(prompt, system_prompt):
    json_response = client.chat.completions.create(
        model='lunartree-gpt-35-turbo-2',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
    ).choices[0].message.content

    print(json_response)

    return json_response

def handle_initial_prompt():
    prompt = st.chat_input("Tell us a little about you.")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        json_system_prompt = type_system_prompt(st.session_state.types)
        type_list = get_json_response(prompt, json_system_prompt)
        relevant_types = json.loads(type_list)["list"]

        st.session_state.relevant_types = relevant_types
        st.session_state.companies = df_ex[df_ex["Company Segment"].isin(relevant_types)].to_dict(orient="records")

        response_system_prompt = company_response_prompt(relevant_types)

        response = client.chat.completions.create(
            model='lunartree-gpt-35-turbo-2',
            messages=[
                {"role": "system", "content": response_system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        
        with st.chat_message("assistant"):
            response = st.write_stream(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.session_state.step = 2
        st.rerun() 

def handle_company_prompt():
    prompt = st.chat_input("Tell us about what type of companies you would like to meet.")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        json_system_prompt = company_system_prompt(st.session_state.companies)
        company_list = get_json_response(prompt, json_system_prompt)
        relevant_companies = json.loads(company_list)["list"]

        st.session_state.relevant_companies = relevant_companies

        response_system_prompt = company_response_prompt(relevant_companies)
        
        response = client.chat.completions.create(
            model='lunartree-gpt-35-turbo-2',
            messages=[
                {"role": "system", "content": response_system_prompt},
            ] + [msg for msg in st.session_state.messages if msg["role"] == "user"],
            stream=True
        )
        
        with st.chat_message("assistant"):
            response = st.write_stream(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        st.session_state.step = 3
        st.rerun() 

def handle_company_details_prompt():
    prompt = st.chat_input("What is your goal at CPHI?")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
        question = "\n ".join([f"User: {msg}" for msg in user_messages])
        
        context_overview = get_similarity_search(question, db_overview)
        context_people = get_similarity_search(question, db_people)
        
        template = get_prompt_template(user_messages, context_overview, context_people)
        print(template)
        
        response = client.chat.completions.create(
            model='lunartree-gpt-35-turbo-2',
            messages=[{"role": "user", "content": template}],
            stream=True
        )

        with st.chat_message("assistant"):
            response = st.write_stream(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


# Show title and description.
st.title("💬 LunarTree CPHI Search Engine")
st.write(
    """This app will find you the right partners to meet at CPHI.
    Please provide some information about who you are and we will highlight the most relevant attendees and organizations."""
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def main():
    if "relevant_types" not in st.session_state:
        st.session_state.relevant_types = []
    
    if "companies" not in st.session_state:
        st.session_state.companies = []

    if "step" not in st.session_state:
        st.session_state.step = 1

    if st.session_state.step == 1:
        handle_initial_prompt()
    if st.session_state.step == 2:
        handle_company_prompt()
    if st.session_state.step == 3:
        handle_company_details_prompt()

if __name__ == "__main__":
    main()