# Create Chinook sqlite database function

def create_chinook_database():
    import sqlite3
    # Connect to the database
    conn = sqlite3.connect("Chinook.db")
    cursor = conn.cursor()
    # Create tables
    with open("Chinook_Sqlite.sql", "r") as sql_file:
        sql_script = sql_file.read()
        cursor.executescript(sql_script)
    # Commit changes and close the connection
    conn.commit()
    conn.close()
    print("Chinook database created successfully.")
    return None
	
-----------------------------------------------------

# Create function to list all table names from sqlite and return as pandas datafram name as name

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
	

-----------------------------------------------------

# Creating a function to call "gemini-1.5-flash" to analyse user input and filter only required
# table list from given list of table as context in prompt

import os
from google.colab import userdata
os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')

import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

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

user_input = "List all artist name with total albums"

prompt_tableList = f"""You are an expert analyst,
Analyse the user_input and table_name Then Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:
{table_names}
Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed.

Here is user input:
{user_input}

Strictly only return list of table_name in pandas list format. No any other text.
"""

def get_table_list_through_model():
  #make connection to gemini-1.5-flash model
  model = genai.GenerativeModel('gemini-1.5-flash')
  response = model.generate_content(prompt_tableList)
  Table_List =[]
  Table_List = response.text
  Table_List = text_to_df(Table_List)
  return Table_List
  

  
----------------------------------------------------------------


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
	

----------------------------------------------------------

# Function to get output from sqlite database with analysis of output and explaination of sql code and output
# using "gemini-1.5-flash" model


# prompt3 working fine and need to use in application

prompt_to_get_sqlwitanalysis = f"""
You are a smart AI assistant to help answer business questions based on analyzing data.
You can plan solving the question with one or multiple thought step. At each thought step, you can write python code to analyze data to assist you. Observe what you get at each step to plan for the next step.

Here is the user question: {user_input}

You are given following utilities to help you retrieve data and communicate your result to end user.
1. execute_sql(sql_query: str): A Python function can query data from the Sqlite3 given a query which you need to create. 
The query has to be syntactically correct for Sqlite3 and only use tables and columns under {filtered_table_list}. 
The execute_sql function returns a Python pandas dataframe contain the results of the query.
5. Don't forget to deal with data quality problem. You should apply data imputation technique to deal with missing data or NAN data.
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
step1_df = execute_sql(sql_query)
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

# function to extract SQL code from model generated text response
import sqlite3
import pandas as pd
conn = sqlite3.connect('Chinook.db')

def extract_sql_from_response(response):
  import re
  SQL_Code = response.text
  sql_code = re.search(r"sql\n(.*?)\n", SQL_Code, re.DOTALL).group(1)
  return sql_code

def execute_sql_query(str,con=conn):
    return pd.read_sql('''{}'''.format(str), con)

def observe(self, name, data):
    try:
        data = data[:5]  # limit the print out observation to 5 elements
    except:
        pass
    print(f"observation:{name}")
    print(data )

import matplotlib.pyplot as plt

def show(data):
    if isinstance(data, plt.Figure):
        plt.show(data)
    else:
        print(data)

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def get_final_output_from_model():
  model = genai.GenerativeModel('gemini-1.5-flash')
  response = model.generate_content(prompt_to_get_sqlwitanalysis)
  SQL_Code = response.text
  return to_markdown(response.text)


-------------------------------------------------

# calling all function in main function

def __main__():
  # create_chinook_database()
  filtered_table_list = get_table_list_through_model()
  table_info=get_table_schema(filtered_table_list)
  final_output = get_final_output_from_model()
  return final_output