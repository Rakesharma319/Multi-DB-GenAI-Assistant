import os
import streamlit as st

print(os.environ['GOOGLE_API_KEY'])
st.write(os.environ['GOOGLE_API_KEY'])

import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

model = genai.GenerativeModel('gemini-1.5-flash')

user_input = st.chat_input("Say something")

response = model.generate_content(user_input)

st.write(to_markdown(response.text))
