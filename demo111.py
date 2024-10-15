import os
import streamlit as st
import google.generativeai as genai

GOOGLE_API_KEY1 = st.chat_input("Paste Google API Key")

genai.configure(api_key=GOOGLE_API_KEY1)

model = genai.GenerativeModel('gemini-1.5-flash')

st.title("_Streamlit_ is :blue[cool] :sunglasses:")
user_input = st.chat_input("Say something")

if user_input:
  response = model.generate_content(user_input)
  st.write(response.text)
else:
  st.write("Pending")
