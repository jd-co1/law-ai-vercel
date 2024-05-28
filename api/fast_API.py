from fastapi import FastAPI,APIRouter
from pydantic import BaseModel
import requests
import streamlit as st
from pymongo import MongoClient
import bson
from llama_index.core.llms import ChatMessage, MessageRole
from datetime import datetime
import json,os
from dotenv import load_dotenv
from typing import Sequence, List
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.agent.openai import OpenAIAgent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
import openai
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import ServiceContext
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core.storage.chat_store import SimpleChatStore
langfuse_callback_handler = LlamaIndexCallbackHandler(
    public_key="pk-lf-063c1421-19ab-4137-acb4-f3c19b7c2fd2",
    secret_key="sk-lf-fd048aa4-44ba-4b2e-9821-3eb388b3e8eb",
    host="https://us.cloud.langfuse.com"
)
load_dotenv()
Settings.callback_manager = CallbackManager([langfuse_callback_handler])
open_ai_key=os.getenv("openai_key")
openai.api_key=open_ai_key
Settings.llm=OpenAI(model="gpt-4o-2024-05-13", temperature=0)
mongo_url=os.getenv("mongo_url")
client = MongoClient(st.secrets["mongo_url"])
mydb = client.get_database('LAW')
information = mydb.law_ai
app = FastAPI()

class Message(BaseModel):
    role: str
    content: str | None = None
   
class chatRequest(BaseModel):
    message: str
    code: str
    columns: List[str]
    id: str
    history: List[Message] | None = None

class Id(BaseModel):
    id: str
def prepare_history(history):
    print("history",history)
    list=[]
    for msg in history:
        print("msg",msg.role)
        list.append(
            ChatMessage(
             role=MessageRole[msg.role.upper()],
             content=msg.content
           )
            )
    return list




def create_history(chat_data,title):
    result = information.insert_one({"name":title,"messages": chat_data,"createdAt": datetime.now()})
    return str(result.inserted_id)

def get_chat_history(chat_id):
    chat_data = information.find_one({"_id": bson.ObjectId(chat_id)})
    # print(chat_data["messages"])
    return chat_data["messages"] if chat_data else None

def fetch_all_chat_ids():
        chat_data = information.find({}, {"name": 1, "_id": 1})

        # Convert results to a list of dictionaries
        all_chat_data = [{"title": doc["name"], "_id": str(doc["_id"])} for doc in chat_data]

        return all_chat_data
    # return [str(doc["_id"]) for doc in information.find({}, {"_id": 1})]

def send_message(message,current_chat_id):
    
    if current_chat_id:
        # print("id:",current_chat_id)
        information.update_one({"_id": bson.ObjectId(current_chat_id)}, {"$push": {"messages": message}})
    else:
        print("id:",current_chat_id)


def get_data(code,column,value):
    'Takes code ,column name, value and returns json response'
    url = f"https://data.cityofnewyork.us/resource/{code}.json?{column}={value}&$limit=5"
    print(url)
    if code=="wvxf-dwi5":
         base_url="https://data.cityofnewyork.us/Housing-Development/Housing-Maintenance-Code-Violations/wvxf-dwi5/explore/query/SELECT*WHERE%20caseless_one_of%28%60{}%60%2C%20%22{}%22%29/page/filter"
         b=base_url.format(column,value)
         st.session_state.url=b
    if code=="6bgk-3dad":
        base_url="https://data.cityofnewyork.us/Housing-Development/DOB-ECB-Violations/6bgk-3dad/explore/query/SELECT*WHERE%20caseless_one_of(%60{}%60,%20%22{}%22)/page/filter"
        b=base_url.format(column,value)
        st.session_state.url=b
    if code=="59kj-x8nc":
        base_url="https://data.cityofnewyork.us/Housing-Development/Housing-Litigations/59kj-x8nc/explore/query/SELECT*WHERE%20caseless_one_of%28%60{}%60%2C%20%22{}%22%29/page/filter"
        b=base_url.format(column,value)
        st.session_state.url=b
    if code=="mdbu-nrqn":
        base_url="https://data.cityofnewyork.us/Housing-Development/Open-Market-Order-OMO-Charges/mdbu-nrqn/explore/query/SELECT*WHERE%20caseless_one_of%28%60{}%60%2C%20%22{}%22%29/page/filter"
        b=base_url.format(column,value)
        st.session_state.url=b
    if code=="tb8q-a3ar":
        base_url="https://data.cityofnewyork.us/Housing-Development/Order-to-Repair-Vacate-Orders/tb8q-a3ar/explore/query/SELECT*WHERE%20caseless_one_of%28%60{}%60%2C%20%22{}%22%29/page/filter"
        b=base_url.format(column,value)
        st.session_state.url=b
    if code=="sbnd-xujn":
        base_url="https://data.cityofnewyork.us/Housing-Development/Handyman-Work-Order-HWO-Charges/sbnd-xujn/explore/query/SELECT*WHERE%20caseless_one_of%28%60{}%60%2C%20%22{}%22%29/page/filter"
        b=base_url.format(column,value)
        st.session_state.url=b
    if code=="ygpa-z7cr":
        base_url="https://data.cityofnewyork.us/Housing-Development/Housing-Maintenance-Code-Complaints-and-Problems/ygpa-z7cr/explore/query/SELECT*WHERE%20caseless_one_of%28%60{}%60%2C%20%22{}%22%29/page/filter"
        b=base_url.format(column,value)
        st.session_state.url=b
    if code=="ipu4-2q9a":
        base_url="https://data.cityofnewyork.us/Housing-Development/DOB-Permit-Issuance/ipu4-2q9a/explore/query/SELECT*WHERE%20caseless_one_of(%60{}%60,%20%22{}%22)/page/filter"
        b=base_url.format(column,value)
        st.session_state.url=b
    if code=="eabe-havv":
        base_url="https://data.cityofnewyork.us/Housing-Development/DOB-Complaints-Received/eabe-havv/explore/query/SELECT*WHERE%20caseless_one_of%28%60{}%60%2C%20%22{}%22%29/page/filter"
        b=base_url.format(column,value)
        st.session_state.url=b
    

    print("url:",st.session_state.url)

    response = requests.get(url)
    return response.json()


def get_location(house_number,street_name,borough):
    'Takes address(house_number,street_name,borough) and returns json response,use this tool when the bin number is not mentioned in the query'
    
    url = "https://api.nyc.gov/geo/geoclient/v1/address.json?houseNumber={}&street={}&borough={}".format(house_number,street_name,borough)

    hdr ={
        # Request headers
        'Cache-Control': 'no-cache',
        'Ocp-Apim-Subscription-Key': '77b7da7befcc4072aa62d8c3b35dc3a0',
        }
    response=requests.get(url,headers=hdr)
    return response.json()

location_tool=FunctionTool.from_defaults(fn=get_location)
nyc_tool=FunctionTool.from_defaults(fn=get_data)
llm = OpenAI(temperature=0, model="gpt-4o-2024-05-13")

chat_store=SimpleChatStore()
memory=ChatMemoryBuffer(token_limit=120000000,chat_store=chat_store,chat_store_key="user1")


system_prompt="""
                This is a data analysis application where users interact with a dataset and external APIs through natural language queries. You are an LLM responsible for understanding user intent, processing queries, and interacting with the APIs.

**Dataset:**

* Schema: {List column names here, e.g., "customer_id", "dob_bis_extract", "address"}
***Use *location_tool* when there  is no bin number explicitly mentioned in the query of user***
*** Use only bin in nyc_tool after you retrieve data from location_tool***
**Tools:**

1. **Data Retrieval tool{nyc_tool}:**
    * Takes a column name and a value as input.
    * Returns data from the external source based on that specific value.
    
2. **Geocoding tool{location_tool}:**
    
    * Takes an address as input.
    * Takes address components (building number, street name, borough) as input.
    * Returns numerical identifiers (BBL=borough block lot, BIN=Building Identification Number or buildingIdentificationNumber) for the location.
    

**User Interaction:**

* Users will provide natural language queries to interact with the data.
* Your task is to interpret their intent and extract relevant information.

**Desired Outputs:**

1. **Data Retrieval:**
    * If the user asks for data based on a column value:
        * Identify the relevant column name (even if user wording slightly differs).
        * Extract the value from the user query.
        * Generate the API call to the Data Retrieval API with the mapped column name and the extracted value.
        * If successful, return the retrieved data along with the corresponding column name for further analysis.
2. **Geocoding:**
    * If the user provides an address:
        * Extract building number, street name, and borough from the address.
        * Generate the API call to the Geocoding API with the extracted address components.
        * If successful, return the BBL and BIN values.
        
3. **Clarification:**
    * If the user's query is unclear or requires additional information (e.g., missing building number), prompt the user for clarification.
4. **important:**
    * column values will not have comma's(,) in them.
    **Always use location_tool when the bin number is not mentioned in the query**
    *** whenever you are using nyc_tool after location_tool, only use the bin number in the query for nyc_tool***
    **If there are five json records in the context,mention it in the response and tell the user to check url for more records.**
                                """


def chat_engine(query,chat_history=[]):

    agent= OpenAIAgent.from_tools(
              [nyc_tool,location_tool], 
              llm=OpenAI(temperature=0, model="gpt-4o-2024-05-13"), 
              verbose=True,
              callback_manager=Settings.callback_manager,
              memory=memory,
              system_prompt=system_prompt
              )
    return agent.chat(query,chat_history)

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/chat")
def chat(request: chatRequest):
    print("request",request)
    if  request.id is None:
        res=llm.complete(f""" from this text:{request.message} give a catogory name and don't mention any other information,just give the name of the catogory and don't mention any other information""")
        print(res)
        
        history=prepare_history(request.history)
        request.history.append({"role":"user","content":(request.message)})

        response = chat_engine(request.message+" "+f"Code:{request.code}"+" Don't mention code and in the response"+" "+f"coulmns: {(request.columns)}",history)
        request.history.append({"role":"assistant","content":response.response})
        request.id=create_history(request.history,res)
        # send_message({"role":"assistant","content":response.response},request.id)

        # return {"status":True,"response":response.response,"history":request.history}
    else:
        history=prepare_history(request.history)
        request.history.append({"role":"user","content":(request.message)})
        send_message({"role":"user","content":(request.message)},request.id)

        response = chat_engine(request.message+" "+f"Code:{request.code}"+" Don't mention code and in the response"+" "+f"coulmns: {(request.columns)}",history)
        request.history.append({"role":"assistant","content":response.response})
        send_message({"role":"assistant","content":response.response},request.id)
    return {"status":True,"response":response.response,"history":request.history,"id":request.id}

@app.get("/ids")
def get_history():
    return fetch_all_chat_ids()

@app.post(f"/history_id")
def get_history(request: Id):
    return get_chat_history(request.id)




