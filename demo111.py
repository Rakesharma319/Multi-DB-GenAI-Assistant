import os
import streamlit as st

print(os.environ['GOOGLE_API_KEY'])

st.title("This is a title")
st.title("_Streamlit_ is :blue[cool] :sunglasses:")
st.write("Hello, *World!* :sunglasses:")
st.write(os.getenv("GOOGLE_API_KEY"))
