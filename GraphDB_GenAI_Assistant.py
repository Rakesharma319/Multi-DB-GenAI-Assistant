import streamlit as st
from langchain_community.graphs import Neo4jGraph
import google.generativeai as genai
from plotly.graph_objects import Figure


import os

st.title("🦜🔗 Graph Database GenAI Assistant")

NEO4J_URI=st.secrets["NEO4J_URI"]
NEO4J_USERNAME=st.secrets["NEO4J_USERNAME"]
NEO4J_PASSWORD=st.sidebar.text_input("Enter NEO4J_PASSWORD",type="password")

graph=Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)
 
schema = graph.schema

# # st.session_state.NEO4J_URI = st.sidebar.text_input("Enter NEO4J_URI")
# st.session_state.NEO4J_URI = "neo4j+s://cc339269.databases.neo4j.io"
# st.session_state.NEO4J_USERNAME = st.sidebar.text_input("Enter NEO4J_USERNAME")
# st.session_state.NEO4J_PASSWORD = st.sidebar.text_input("Enter NEO4J_PASSWORD",type="password")

os.environ["NEO4J_URI"]=NEO4J_URI
os.environ["NEO4J_USERNAME"]=NEO4J_USERNAME
os.environ["NEO4J_PASSWORD"]=NEO4J_PASSWORD


if schema:
    st.write("Database Connection Success!!")
else:
    st.write("Check DB Connection")
    
question=st.text_input("Ask Question")

google_api_key = st.sidebar.text_input('Google API Key', type='password')

# function to display llm responce
def display_output(llm_response):
  import pandas as pd
  import matplotlib.pyplot as plt
  import numpy as np

  # def execute_cypher_query(cyphercode):
  #   import pandas as pd
  #   DFrmae=pd.DataFrame(graph.query(cyphercode))
  #   return DFrmae
  
  def execute_cypher_query(cyphercode):
    import pandas as pd
    uri = NEO4J_URI
    username = NEO4J_USERNAME
    password = NEO4J_PASSWORD
    graph=Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,)
    Pd_df = pd.DataFrame(graph.query(cyphercode))
    return Pd_df
  
  def observe(name, data):
    try:
      data = data[:5]  # limit the print out observation to 5 elements
    except:
      pass
    st.write(f"observation:{name}")
    st.write(data )

  def show(data):
    if type(data) is Figure:
      st.plotly_chart(data)
    else:
      st.write(data)

  actions = llm_response.split("```")[1::2]

  python_actions = [action for action in actions if "python" in action]
  python_code1 = python_actions[0]
  python_code2 = python_actions[1]

  python_code1 = python_code1.replace("python\n","")
  python_code2 = python_code2.replace("python\n","")

  actions3 = llm_response.split('\n')
  for action3 in actions3:
    if "Question" in action3:
      st.write(action3,"\n")

    if "Thought 1" in action3:
      st.write(action3)
      st.code(python_code1,language="python")
      exec(python_code1,locals())

    if "Thought 2" in action3:
      st.write(action3)
      st.code(python_code2,language="python")
      exec(python_code2,locals())

    if "Answer" in action3:
      st.write(action3)


CYPHER_GENERATION_TEMPLATE = f"""
You are a smart AI assistant to help answer business questions based on analyzing data.
You can plan solving the question with one or multiple thought step. 
At each thought step, you can write python code to analyze data to assist you. Observe what you get at each step to plan for the next step.

Here is the user question: {question}

You are given following utilities to help you retrieve data and communicate your result to end user.
1. execute_cypher_query(cypher_query: str): A Python function can query data from the Neo4j given a query which you need to create. 
The query has to be syntactically correct for Neo4j and Use only the provided relationship types and properties in the schema.
Use alias when using the WITH keyword in cypher query.
Do not use any other relationship types or properties that are not provided.
Schema:{schema}. The execute_cypher_query function returns a Python pandas dataframe contain the results of the query.
2. Use plotly library for data visualization.
    - When ever you try to read column data from the panadas dataframe, use column name as m.col_name instead only col_name.
    - Example: fig = px.bar(step1_df, x="m.title", y="m.imdbRating", title="Top 10 Movies by IMDb Ratings")
3. Use observe(label: str, data: any) utility function to observe data under the label for your evaluation. 
Use observe() function instead of print() as this is executed in streamlit environment. 
Due to system limitation, you will only see the first 10 rows of the dataset.
4. To communicate with user, use show() function on data, text and plotly figure. show() is a utility function that can render different types of data to end user. Remember, you don't see data with show(), only user does. You see data with observe()
    - If you want to show  user a plotly visualization, then use ```show(fig)```
    - If you want to show user data which is a text or a pandas dataframe or a list, use ```show(data)```
    - Never use print(). User don't see anything with print()
5. Lastly, don't forget to deal with data quality problem. You should apply data imputation technique to deal with missing data or NAN data.
6. Always follow the flow of Thought: , Observation:, Action: and Answer: as in template below strictly.

<<Template>>
Question: User Question
Thought 1: Your thought here.
Action:
```python
#Import neccessary libraries here
import numpy as np
#Query some data
cypher_query = "SOME CYPHER QUERY"
step1_df = execute_cypher_query(cypher_query)
# Replace 0 with NaN. Always have this step
step1_df['Some_Column'] = step1_df['Some_Column'].replace(0, np.nan)

#observe query result
```
Observation:
step1_df is displayed here
Thought 2: Your thought here
Action:
```python
import plotly.express as px
#from step1_df, perform some data analysis action to produce step2_df
#To see the data for yourself the only way is to use observe()
observe("some_label", step2_df) #Always use observe()
#Decide to show it to user, Use alias when using the WITH keyword
fig=px.line(step2_df)
#visualize fig object to user.
show(fig)
#you can also directly display tabular or text data to end user.
show(step2_df)
```
Observation:
step2_df is displayed here
Answer: Your final answer and comment for the question
<</Template>>
"""

# st.write(CYPHER_GENERATION_TEMPLATE)
def graphDB_main(CYPHER_GENERATION_TEMPLATE):
    genai.configure(api_key = google_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(CYPHER_GENERATION_TEMPLATE)
    llm_response = response.text
    st.write(llm_response)
    # display_output(llm_response)

# graphDB_main()
