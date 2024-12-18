import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore
import streamlit as st
import os

def get_final_output_from_model(prompt_template):
  model = genai.GenerativeModel('gemini-1.5-flash')
  response = model.generate_content(prompt_template)
  SQL_Code = response.text
  return response.text

# Streamlit App 

st.markdown(
    """# **Vector Database Gemini Assistant**
This is an experimental assistant that requires OpenAI access. The app demonstrates the use of Gemini AI to support getting insights from Vector Database by just asking questions. 
This assistant has RAG technique used to get more accurate response out of similirity search output of vector db.
"""
)
col1, col2 = st.columns((3, 1))

with st.sidebar:
    astradb_api_key = st.text_input('Astra DB API Key', type='password')
    google_api_key = st.text_input('Google API Key', type='password')
    question = st.text_area("Ask me a question")
    
    if question:
        os.environ["ASTRA_DB_API_ENDPOINT"] ="https://5e5c552b-3a72-4b4b-bd83-0e2e0f12347a-us-east-2.apps.astra.datastax.com"
        os.environ["ASTRADB_API_KEY"] =astradb_api_key
        os.environ["GOOGLE_API_KEY"] = google_api_key
        
        genai.configure(api_key=google_api_key)
        # Configure your embedding model and vector store
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        vstore = AstraDBVectorStore(
            collection_name="qa_mini_demo2",
            embedding=embedding,
            token=os.getenv("ASTRADB_API_KEY"),
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        )

        retriever = vstore.as_retriever(search_kwargs={"k": 3})
        retriver_op = retriever.invoke(question)
        
        prompt_template = f"""Answer the question based only on the supplied context. If you don't know the answer, say you don't know the answer.
        If you know the answer then give your openion on the final result in 100 words, and tag it with 'My Openion' followed by original result.
        Context: {retriver_op}
        Question: {question}
        Your answer:
        Your Openion :
        """
        
        final_output = get_final_output_from_model(prompt_template)
        
        col1.write("Astra vector store configured")
        col1.write("User Question:")
        col1.write(question)
        col1.write("Response:")
        col1.write(final_output)
        col1.write("similar search output")
        col1.write(retriver_op)

    else:
        st.write("Please ask question !!")
