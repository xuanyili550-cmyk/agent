import os
import openai
import sys
sys.path.append('../..')
from dotenv import load_dotenv, find_dotenv
_=load_dotenv(find_dotenv())
openai.api_key  = os.environ['OPENAI_API_KEY']
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()
print(len(pages))
page=pages[0]
print(page)
print(page.page_content[0:500])
print(page.metadata)
from langchain_community.document_loaders.generic import GenericLoader,FileSystemBlobLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
save_dir="docs/youtube/"
loader=GenericLoader(
    FileSystemBlobLoader(save_dir,glob="*.m4a"),
    OpenAIWhisperParser()
)
docs=loader.load()
print(docs[0].page_content[0:500])
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/titles-for-programmers.md")
docs = loader.load()
print(docs[0].page_content[:500])
from langchain_community.document_loaders import NotionDirectoryLoader
loader=NotionDirectoryLoader("docs/Notion_DB")
docs=loader.load()
print(docs[0].page_content[0:200])
print(docs[0].metadata)
