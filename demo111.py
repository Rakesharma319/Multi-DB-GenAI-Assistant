import os
import streamlit as st

GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']

import google.generativeai as genai
import os

genai.configure(api_key=os.environ["API_KEY"])


model = genai.GenerativeModel('gemini-1.5-flash')

user_input = st.chat_input("Say something")

response = model.generate_content(user_input)

st.write(response.text)
