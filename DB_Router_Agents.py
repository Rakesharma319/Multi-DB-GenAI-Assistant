import streamlit as st
from RDBMS_GenAI_Assistant import *
from GraphDB_GenAI_Assistant import *

# Streamlit app
st.title("🦜🔗 Database Router Agents GenAI Assistant")
col1, col2 = st.columns((3, 1))

google_api_key = st.sidebar.text_input('Google API Key', type='password')
NEO4J_PASSWORD=st.sidebar.text_input("Enter NEO4J_PASSWORD",type="password")

# User input for name
question = st.sidebar.text_area("Ask me a question")


# Dropdown for function selection
function_choice = st.sidebar.selectbox("Choose a function to call", ["Relational_Database", "Graph_Database"])

# Button to call the function
if st.sidebar.button("Call Function"):
    if function_choice == "Relational_Database":
        # rdbms_main()
        response = rdbms_main(prompt_to_get_sqlwitanalysis)
        col1.write("Relational_Database")
        # col1.write(response)
        display_output(response)
    elif function_choice == "Graph_Database":
        gdb_response = graphDB_main(CYPHER_GENERATION_TEMPLATE,google_api_key)
        col1.write("Graph_Database")
        # col1.write(gdb_response)
        display_output(gdb_response)
    else:
        result = "Invalid choice"

