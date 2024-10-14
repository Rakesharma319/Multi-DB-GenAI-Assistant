import os
import streamlit as st
import google.generativeai as genai

genai.configure(api_key=sys.argv[1])

model = genai.GenerativeModel('gemini-1.5-flash')

st.title("_Streamlit_ is :blue[cool] :sunglasses:")
user_input = st.chat_input("Say something")

response = model.generate_content(user_input)

st.write(response.text)
