import os
import streamlit as st

from dotenv import load_dotenv
import os

load_dotenv()  # This line brings all environment variables from .env into os.environ
print(os.environ['GOOGLE_API_KEY'])

st.title("This is a title")
st.title("_Streamlit_ is :blue[cool] :sunglasses:")
st.write("Hello, *World!* :sunglasses:")
st.write(os.getenv("GOOGLE_API_KEY"))
