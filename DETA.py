from dotenv import load_dotenv
import streamlit as st
#import docx
import pandas as pd
from transformers import pipeline
import os
import boto3 
from io import BytesIO

load_dotenv()
region_name = os.getenv('region_name')
api_key = os.getenv('aws_access_key_id')
secret_key = os.getenv('aws_secret_access_key')

session = boto3.Session(
    aws_access_key_id=api_key,
    aws_secret_access_key=secret_key,
    region_name=region_name
)
print(session)
client = session.client('s3')  # Example: Replace 's3' with your AWS service

# s3_client= session.client('s3')

response = client.get_object(Bucket="detax",Key='DETA DATASET.xlsx')
# #s3://detax/DETA DATASET.xlsx
# #response = client.get_object(Bucket="detax",Key='detax/DETA DATASET.xlsx')
content= response['Body']#.decode('utf-8')
data = content.read()

# Use BytesIO to simulate a file object
file_like_object = BytesIO(data)


def main():
    
    st.set_page_config(page_title="Diet")
    st.header("Diet Recipe Ideas  💬")


    
    df = pd.read_excel(file_like_object)

    try:
        user_question = st.text_input("Ask for Recipe")

        qa_model = pipeline("table-question-answering",model='google/tapas-large-finetuned-wtq')#tokenizer='google-bert/bert-large-uncased-whole-word-masking-finetuned-squad')
        question = user_question
        context = df
        answer=qa_model(table=df,query = question)
        
        st.write(answer['answer'])
    except ValueError:
        st.write("Ask for recipe")

        

if __name__ == '__main__':
    main()