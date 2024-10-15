import os
import streamlit as st
import google.generativeai as genai

google_api_key = st.sidebar.text_input('Google API Key', type='password')

genai.configure(api_key=google_api_key)

max_response_tokens = 1500
token_limit = 6000
temperature = 0.2

Enter = "False"
def runapp():
    Enter = "True"

st.markdown(
    """# **Database Gen AI Assistant**
This is an experimental assistant that requires OpenAI access. The app demonstrates the use of OpenAI to support getting insights from Database by just asking questions. The assistant can also generate SQL and Python code for the Questions.
"""
)

model = genai.GenerativeModel('gemini-1.5-flash')
user_input = st.sidebar.text_area("Ask me a question")

if user_input:
  response = model.generate_content(user_input)
  st.write(response.text)
else:
  st.write("Pending")
