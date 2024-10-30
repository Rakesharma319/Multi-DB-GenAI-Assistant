import streamlit as st
from RDBMS_GenAI_Assistant import rdbms_main
from GraphDB_GenAI_Assistant import graphDB_main

# Streamlit app
st.title("🦜🔗 Database Router Agents GenAI Assistant")

# User input for name
name = st.text_input("Enter your DB")

# Dropdown for function selection
function_choice = st.selectbox("Choose a function to call", ["Relational_Database", "Graph_Database"])

# Button to call the function
if st.button("Call Function"):
    if function_choice == "Relational_Database":
        rdbms_main()
    elif function_choice == "Graph_Database":
        graphDB_main()
    else:
        result = "Invalid choice"


# Run the Streamlit app
if __name__ == "__main__":
    st.run()
