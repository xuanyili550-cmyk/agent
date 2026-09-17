import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
import warnings
warnings.filterwarnings('ignore')

import datetime
current_data=datetime.datetime.now().date()
target_data=datetime.date(2024,6,12)

if current_data>target_data:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain_openai import OpenAI

file = 'OutdoorClothingCatalog_1000.csv'
loader=CSVLoader(file_path=file)
from langchain_classic.indexes import VectorstoreIndexCreator

from langchain_openai import OpenAIEmbeddings

index=VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=OpenAIEmbeddings(),
).from_loaders([loader])
query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."

llm_replacement_model = OpenAI(temperature=0,
                               model='gpt-3.5-turbo-instruct')

response=index.query(query,llm=llm_replacement_model)
print(display(Markdown(response)))
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader(file_path=file)

docs = loader.load()
embeddings = OpenAIEmbeddings()
embed = embeddings.embed_query("Hi my name is Harrison")
print(len(embed))
print(embed[:5])
db = DocArrayInMemorySearch.from_documents(
    docs,
    embeddings
)

query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)
print(len(docs))
retriever = db.as_retriever()
llm = ChatOpenAI(temperature = 0.0, model=llm_model)
qdocs = "".join([docs[i].page_content for i in range(len(docs))])
response = llm.invoke(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.").content
print(display(Markdown(response)))
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)

query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."
response = qa_stuff.run(query)
display(Markdown(response))
response = index.query(query, llm=llm)
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])