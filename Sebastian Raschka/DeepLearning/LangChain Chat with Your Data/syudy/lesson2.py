import os
import openai
import sys
sys.path.append('../..')
from dotenv import load_dotenv,find_dotenv
_=load_dotenv(find_dotenv())
openai.api_key=os.environ['OPENAI_API_KEY']
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter

chunk_size=26
chunk_overlap=4

r_splitter=RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
c_splitter=CharacterTextSplitter(
    chunk_overlap=chunk_overlap,
    chunk_size=chunk_size
)
text1 = 'abcdefghijklmnopqrstuvwxyz'
print(r_splitter.split_text(text1))
text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
print(r_splitter.split_text(text2))
text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
print(r_splitter.split_text(text3))
print(c_splitter.split_text(text3))
c_splitter=CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separator=" "
)
print(c_splitter.split_text(text3))
some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""
c_splitter = CharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    separator = ' '
)
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", ""]
)
print(c_splitter.split_text(some_text))
print(r_splitter.split_text(some_text))
r_splitter=RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "\. ", " ", ""],
    is_separator_regex=True,
)
print(r_splitter.split_text(some_text))
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""],
    is_separator_regex=True,
)
print(r_splitter.split_text(some_text))

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()
from langchain_text_splitters import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)
docs=text_splitter.split_documents(pages)
print(len(docs))
print(len(pages))

from langchain_community.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("docs/Notion_DB")
notion_db = loader.load()
docs=text_splitter.split_documents(notion_db)
print(len(docs))
print(len(notion_db))
from langchain_text_splitters import TokenTextSplitter
text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
text1 = "foo bar bazzyfoo"
print(text_splitter.split_text(text1))
text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
docs = text_splitter.split_documents(pages)
print(pages[0].metadata)

from langchain_community.document_loaders import NotionDirectoryLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_document = """# Title\n\n \
## Chapter 1\n\n \
Hi this is Jim\n\n Hi this is Joe\n\n \
### Section \n\n \
Hi this is Lance \n\n
## Chapter 2\n\n \
Hi this is Molly"""
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
md_header_splits = markdown_splitter.split_text(markdown_document)
print(md_header_splits[0])
print(md_header_splits[1])
loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()
txt = ' '.join([d.page_content for d in docs])

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
md_header_splits = markdown_splitter.split_text(txt)
print(md_header_splits[0])



