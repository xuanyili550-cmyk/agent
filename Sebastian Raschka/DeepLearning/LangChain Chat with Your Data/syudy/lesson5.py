import os
import openai
import sys
sys.path.append('../..')
from dotenv import load_dotenv,find_dotenv
_=load_dotenv(find_dotenv())
openai.api_key=os.environ['OPENAI_API_KEY']
import datetime
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

persist_directory = 'docs/chroma/'
embedding=OpenAIEmbeddings()
vectordb=Chroma(persist_directory=persist_directory,embedding_function=embedding)
print(vectordb._collection.count())
question = "What are major topics for this class?"
docs = vectordb.similarity_search(question,k=3)
print(len(docs))

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0)

from langchain_classic.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

result = qa_chain({"query": question})

print(result["result"])
from langchain_core.prompts import PromptTemplate

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT=PromptTemplate.format_prompt(template)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
question = "Is probability a class topic?"
result = qa_chain({"query": question})
print(result["source_documents"][0])
print(result["result"])
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)

result = qa_chain_mr({"query": question})
print(result["result"])
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)
result = qa_chain_mr({"query": question})
print(result["result"])
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="refine"
)
result = qa_chain_mr({"query": question})
print(result["result"])
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)
question = "Is probability a class topic?"
result = qa_chain({"query": question})
print(result["result"])
question = "why are those prerequesites needed?"
result = qa_chain({"query": question})
print(result["result"])
