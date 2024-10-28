import streamlit as st
import os

st.title("🦜🔗 Quickstart App")

NEO4J_URI = st.text_input("Enter NEO4J_URI")
NEO4J_USERNAME = st.text_input("Enter NEO4J_USERNAME")
NEO4J_PASSWORD = st.text_input("Enter NEO4J_PASSWORD",type="password")

st.write(NEO4J_URI)
st.write(NEO4J_USERNAME)

os.environ["NEO4J_URI"]=NEO4J_URI
os.environ["NEO4J_USERNAME"]=NEO4J_USERNAME
os.environ["NEO4J_PASSWORD"]=NEO4J_PASSWORD

if st.button("Login"):
    from langchain_community.graphs import Neo4jGraph
    graph=Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )
    
    schema = graph.schema
    
    if schema:
        st.write("Database Connection Success!!")
    else:
        st.write("Check DB Connection")
