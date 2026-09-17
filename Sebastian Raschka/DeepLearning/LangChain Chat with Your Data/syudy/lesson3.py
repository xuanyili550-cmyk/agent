import os
import openai
import sys
sys.path.append('../..')
from dotenv import load_dotenv,find_dotenv
_=load_dotenv(find_dotenv())
openai.api_key=os.environ['OPENAI_API_KEY']
from langchain_community.document_loaders import PyPDFLoader
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture03.pdf")
]
docs=[]
for loader in loaders:
    docs.extend(loader.load())
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)
splits=text_splitter.split_documents(docs)
print(len(splits))
from langchain_openai import OpenAIEmbeddings
embedding=OpenAIEmbeddings()
sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"
embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)
import numpy as np
np.dot(embedding1,embedding2)
np.dot(embedding1,embedding3)
np.dot(embedding2,embedding3)
from langchain_community.vectorstores import Chroma
persist_directory = 'docs/chroma/'
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
print(vectordb._collection.count())
question = "is there an email i can ask for help"
docs=vectordb.similarity_search(question,k=3)
print(len(docs))
print(docs[0].page_content)
print(vectordb.persist)

question = "what did they say about matlab?"
docs = vectordb.similarity_search(question,k=5)
question = "what did they say about regression in the third lecture?"
docs = vectordb.similarity_search(question,k=5)
for doc in docs:
    print(doc.metadata)
print(docs[4].page_content)