from RDBMS_GenAI_Assistant import rdbms_main
from GraphDB_GenAI_Assistant import GraphDB_main
import streamlit as st

st.title("🦜🔗 Database Router Agents GenAI Assistant")

genre = st.radio(
    "Choose Databse to Chat",
    ["***Relational_Database***", "***Graph_Database***"],
    captions=[
        "Relational DB contains Music Related data.",
        "Graph DB contains Movie related data.",
    ],
)

if genre == "Relational_Database":
    st.write("You selected Relational Database.")
    rdbms_main()
elif genre == "Graph_Database":
    st.write("You selected Graph Database.")
    GraphDB_main()
else:
    st.write("Not yet Implemented.")
