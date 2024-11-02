import streamlit as st
from RDBMS_GenAI_Assistant import *
from GraphDB_GenAI_Assistant import *

# Streamlit app
st.title("🦜🔗 Database Router Agents GenAI Assistant")

google_api_key = st.sidebar.text_input('Google API Key', type='password')
NEO4J_PASSWORD=st.sidebar.text_input("Enter NEO4J_PASSWORD",type="password")

# User input for name
name = st.text_input("Enter your DB")

# Dropdown for function selection
function_choice = st.selectbox("Choose a function to call", ["Relational_Database", "Graph_Database"])

# Button to call the function
if st.button("Call Function"):
    if function_choice == "Relational_Database":
        # rdbms_main()
        print("Relational_Database")
    elif function_choice == "Graph_Database":
        # graphDB_main()
        print("Graph_Database")
    else:
        result = "Invalid choice"


# Run the Streamlit app
if __name__ == "__main__":
    st.run()
