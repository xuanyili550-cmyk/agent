import os
import openai
import sys
sys.path.append('../..')
from dotenv import load_dotenv,find_dotenv
_=load_dotenv(find_dotenv())
openai.api_key=os.environ['OPENAI_API_KEY']
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'
embedding=OpenAIEmbeddings()
vectordb=Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)
print(vectordb._collection.count())
texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]
smalldb=vectordb.from_texts(texts,embedding=embedding)
question = "Tell me about all-white mushrooms with large fruiting bodies"
smalldb.similarity_search(question,k=2)
smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3)
question = "what did they say about matlab?"
docs_ss = vectordb.similarity_search(question,k=3)
print(docs_ss[1].page_content[:100])
docs_mmr = vectordb.max_marginal_relevance_search(question,k=3)
print(docs_mmr[1].page_content[:100])
question = "what did they say about regression in the third lecture?"
docs = vectordb.similarity_search(
    question,
    k=3,
    filter={"source":"docs/cs229_lectures/MachineLearning-Lecture03.pdf"}
)
for d in docs:
    print(d.metadata)

from langchain_openai import OpenAI
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.base import AttributeInfo

metadata_field_info=[
    AttributeInfo(
        name="source",
        description="The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the lecture",
        type="integer",
    ),
]

from langchain_community.query_constructors.chroma import ChromaTranslator
document_content_description = "Lecture notes"
llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True,
    structured_query_translator=ChromaTranslator(),
)
question = "what did they say about regression in the third lecture?"
docs = retriever.invoke(question)
for d in docs:
    print(d.metadata)
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever()
)
question = "what did they say about matlab?"
compressed_docs = compression_retriever.invoke(question)
pretty_print_docs(compressed_docs)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type = "mmr")
)

question = "what did they say about matlab?"
compressed_docs = compression_retriever.invoke(question)
pretty_print_docs(compressed_docs)
from langchain_community.retrievers import SVMRetriever
from langchain_community.retrievers import TFIDFRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()
all_page_text=[p.page_content for p in pages]
joined_page_text=" ".join(all_page_text)
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=150)
splits=text_splitter.split_text(joined_page_text)
svm_retriever=SVMRetriever.from_texts(splits,embeddings=embedding)
tfidf_retriever=TFIDFRetriever.from_texts(splits)

question = "What are major topics for this class?"
docs_svm=svm_retriever.invoke(question)
docs_svm[0]

question = "what did they say about matlab?"
docs_tfidf=tfidf_retriever.invoke(question)
docs_tfidf[0]