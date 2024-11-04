import streamlit as st

import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from plotly.graph_objs import Figure
import sqlite3
import pandas as pd
conn = sqlite3.connect('Chinook.db')

from langchain_community.graphs import Neo4jGraph
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore


#---- RDBMS Function -----------
def rdbms_main(google_api_key,user_input,st):
    def get_all_table_names():
        import sqlite3
        import pandas as pd
        # Connect to the database
        conn = sqlite3.connect("Chinook.db")
        cursor = conn.cursor()
        sql = '''select distinct name
            from sqlite_master
            where type='table';'''
        cursor.execute(sql)
        tables_List = cursor.fetchall()
        df = pd.DataFrame(tables_List)
        df.columns = ['name']
        return df
    
    def text_to_df(Table_List):
        import re
        import ast
        import pandas as pd
        text = Table_List
        # Extract the list using regular expressions
        list_str = re.search(r"\[.*\]", text).group()
        # Safely evaluate the string as a Python literal
        data_list = ast.literal_eval(list_str)
        # Convert the list to a DataFrame (if needed)
        df = pd.DataFrame(data_list, columns=['name'])
        return df
    
    def get_table_list_through_model():
        #make connection to gemini-1.5-flash model
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt_tableList)
        Table_List =[]
        Table_List = response.text
        Table_List = text_to_df(Table_List)
        return Table_List
    
    # Function to get filtered table schemas details by passing table lists
    def get_table_schema(tables_List):
        import sqlite3
        import pandas as pd
        conn = sqlite3.connect('Chinook.db')
        c = conn.cursor()
        def sq(str,con=conn):
            return pd.read_sql('''{}'''.format(str), con)
        output=[]
        # Initialize variables to store table and column information
        current_table = ""
        columns = []
        for index,row in tables_List.iterrows():
            #print(row["name"])
            #table_schema="DWH"
            table_name_single=row["name"]
            # table_name = f"{table_schema}.{table_name_single}"
            df = sq(f'''PRAGMA table_info({table_name_single});''',conn)
            for index,row in df.iterrows():
                # table_name = f"{table_schema}.{table_name_single}"
                table_name = f"{table_name_single}"
                column_name = row["name"]
                data_type = row["type"]
                if " " in table_name:
                    table_name = f"[{table_name}]"
                    column_name = row["name"]
                if " " in column_name:
                    column_name = row["name"]
    
                # If the table name has changed, output the previous table's information
                if current_table != table_name and current_table != "":
                    output.append(f"table: {current_table}, columns: {', '.join(columns)}")
                    columns = []
    
                # Add the current column information to the list of columns for the current table
                columns.append(f"{column_name} {data_type}")
    
                # Update the current table name
                current_table = table_name
    
        # Output the last table's information
        output.append(f"table: {current_table}, columns: {', '.join(columns)}")
        output = "\n".join(output)
        return output
    
    def get_final_output_from_model():
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt_to_get_sqlwitanalysis)
        SQL_Code = response.text
        return response.text
    
    # function to display llm responce
    
    def display_output(responce,st):
        import sqlite3
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
    
        conn = sqlite3.connect('Chinook.db')
        def execute_sql_query(str,con=conn):
            return pd.read_sql('''{}'''.format(str), con)
        
        def observe(name, data):
            try:
                data = data[:10]  # limit the print out observation to 5 elements
            except:
                pass
            st.write(f"observation:{name}")
            st.write(data )
        
        def show(data):
            if type(data) is Figure:
                st.plotly_chart(data)
            else:
                st.write(data)
        
        actions = responce.split("```")[1::2]
        
        python_actions = [action for action in actions if "python" in action]
        python_code1 = python_actions[0]
        python_code2 = python_actions[1]
        
        python_code1 = python_code1.replace("python\n","")
        python_code2 = python_code2.replace("python\n","")
        
        actions3 = responce.split('\n')
        for action3 in actions3:
            if "Question" in action3:
                st.write(action3)
        
            if "Thought 1" in action3:
                st.write(action3)
                # st.write(python_code1)
                st.code(python_code1,language="python")
                exec(python_code1,locals())
            # print("\n")
            
            if "Thought 2" in action3:
                st.write(action3)
                # st.write(python_code2)
                st.code(python_code2,language="python")
                exec(python_code2,locals())
            # print("\n")
            
            if "Answer" in action3:
                st.write(action3)
    # ------------- Streamlit app --------------
    
    #-------------- Ask api key
    #google_api_key = st.sidebar.text_input('Google API Key', type='password')
    genai.configure(api_key = google_api_key)
    
    #user_input = st.sidebar.text_area("Ask me a question")
    
    table_names = get_all_table_names()
    
    prompt_tableList = f"""You are an expert analyst,
    Analyse the user_input and table_name Then Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
    The tables are:
    {table_names}
    Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed.
    
    Here is user input:
    {user_input}
    
    Strictly only return list of table_name in pandas list format. No any other text.
    """
    
    filtered_table_list = get_table_list_through_model()
    table_info=get_table_schema(filtered_table_list)
    
    prompt_to_get_sqlwitanalysis = f"""
    You are a smart AI assistant to help answer business questions based on analyzing data.
    You can plan solving the question with one or multiple thought step. At each thought step, you can write python code to analyze data to assist you. Observe what you get at each step to plan for the next step.
    
    Here is the user question: {user_input}
    
        You are given following utilities to help you retrieve data and communicate your result to end user.
        1. execute_sql_query(sql_query: str): A Python function can query data from the Sqlite3 given a query which you need to create. The query has to be syntactically correct for Sqlite3 and 
            only use tables and columns under {table_info}. The execute_sql function returns a Python pandas dataframe contain the results of the query.
        2. Use plotly library for data visualization.
        3. Use observe(label: str, data: any) utility function to observe data under the label for your evaluation. Use observe() function instead of print() as this is executed in streamlit environment. Due to system limitation, you will only see the first 10 rows of the dataset.
        4. To communicate with user, use show() function on data, text and plotly figure. show() is a utility function that can render different types of data to end user. Remember, you don't see data with show(), only user does. You see data with observe()
            - If you want to show  user a plotly visualization, then use ```show(fig)`` 
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
    sql_query = "SOME SQL QUERY"
    step1_df = execute_sql_query(sql_query)
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
    #Decide to show it to user.
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
    response = get_final_output_from_model()
    display_output(response,st)
#----- RDBMS End -------------

#-------- Grapg DB Funation ---------
def graphDB_main(NEO4J_PASSWORD,google_api_key,question,st):
    NEO4J_URI=st.secrets["NEO4J_URI"]
    NEO4J_USERNAME=st.secrets["NEO4J_USERNAME"]
    
    graph=Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )
    
    schema = graph.schema
    
    os.environ["NEO4J_URI"]=NEO4J_URI
    os.environ["NEO4J_USERNAME"]=NEO4J_USERNAME
    os.environ["NEO4J_PASSWORD"]=NEO4J_PASSWORD
    
    if schema:
        st.write("Database Connection Success!!")
    else:
        st.write("Check DB Connection")
    
    # function to display llm responce
    def display_output(llm_response,st):
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
    
    genai.configure(api_key = google_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(CYPHER_GENERATION_TEMPLATE)
    llm_response = response.text
    display_output(llm_response,st)

#-------- Graph DB end ------------

# ---------------- vector db main function -----------

def astradb_main_funct(ASTRADB_API_KEY,google_api_key,question,st):
    
    def get_final_output_from_model():
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt_template)
        SQL_Code = response.text
        return response.text
    
    # Streamlit App 
    st, col2 = st.columns((3, 1))
    st.markdown(
        """# **Vector Database Gemini Assistant**
    This is an experimental assistant that requires OpenAI access. The app demonstrates the use of Gemini AI to support getting insights from Vector Database by just asking questions. 
    This assistant has RAG technique used to get more accurate response out of similirity search output of vector db.
    """
    )

    
    if question:
        os.environ["ASTRA_DB_API_ENDPOINT"] ="https://5e5c552b-3a72-4b4b-bd83-0e2e0f12347a-us-east-2.apps.astra.datastax.com"
        os.environ["ASTRADB_API_KEY"] =ASTRADB_API_KEY
        os.environ["google_api_key"] = google_api_key
        
        genai.configure(api_key=google_api_key)
        # Configure your embedding model and vector store
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        vstore = AstraDBVectorStore(
            collection_name="qa_mini_demo2",
            embedding=embedding,
            token=os.getenv("ASTRADB_API_KEY"),
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        )
    
        retriever = vstore.as_retriever(search_kwargs={"k": 3})
        retriver_op = retriever.invoke(question)
        
        prompt_template = f"""Answer the question based only on the supplied context. If you don't know the answer, say you don't know the answer.
        If you know the answer then give your openion on the final result in 100 words, and tag it with 'My Openion' followed by original result.
        Context: {retriver_op}
        Question: {question}
        Your answer:
        Your Openion :
        """
        
        final_output = get_final_output_from_model()
        
        st.write("Astra vector store configured")
        st.write("User Question:")
        st.write(question)
        st.write("Response:")
        st.write(final_output)
        st.write("similar search output")
        st.write(retriver_op)
    
    else:
        st.write("Please ask question !!")

#------ Astra DB end --------------------------

# Stramlit app
col1, col2 = st.columns((3, 1))
st.title("🦜🔗 Database Router Agents GenAI Assistant")

with st.sidebar:
    ASTRADB_API_KEY = st.text_input('Astra DB API Key', type='password')
    NEO4J_PASSWORD = st.text_input('Enter NEO4J_PASSWORD', type='password')
    google_api_key = st.text_input('Google API Key', type='password')
    question = st.text_area("Ask me a question")
    # Dropdown for function selection
    function_choice = st.selectbox("Choose a function to call", ["Relational_Database", "Graph_Database","Vector_Database"])

    # Button to call the function
    if st.sidebar.button("Call Function"):
        if function_choice == "Relational_Database":
            response = rdbms_main(google_api_key,question,col1)
            st.write("Relational_Database")
            st.write(response)
        elif function_choice == "Graph_Database":
            gdb_response = graphDB_main(NEO4J_PASSWORD,google_api_key,question,col1)
            st.write("Graph_Database")
            st.write(gdb_response)
        elif function_choice == "Vector_Database":
            vdb_response = astradb_main_funct(ASTRADB_API_KEY,google_api_key,question,col1)
            st.write("Vector_Database")
            st.write(vdb_response)
        else:
            result = "Invalid choice"

