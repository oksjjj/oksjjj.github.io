---
title: Ch#7 - BUILDING LLMS FOR PRODUCTION 코드 현행화
author: oksjjj
date: 2025-04-14 19:00:00 +0900
categories: [LLM/RAG]
tags: [LLM, RAG, langchain, chat, gpt, llama, index, deeplake]
render_with_liquid: false
---

### python 버전

3.9

```python
from dotenv import load_dotenv
import os
load_dotenv('../env')

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['ACTIVELOOP_TOKEN'] = os.getenv('ACTIVELOOP_TOKEN')
```

### RetrievalQA

```python
from langchain.document_loaders import TextLoader

text =""" Google opens up its AI language model PaLM to challenge OpenAI and GPT-3
Google offers developers access to one of its most advanced AI language models: PaLM.
The search giant is launching an API for PaLM alongside a number of AI enterprise tools
it says will help businesses "generate text, images, code, videos, audio,
and more from simple natural language prompts."

PaLM is a large language model, or LLM, similar to the GPT series created by OpenAI
or Meta's LLaMA family of models. Google first announced PaLM in April 2022.
Like other LLMs, PaLM is a flexible system that can potentially carry out
all sorts of text generation and editing tasks.
You could train PaLM to be a conversational chatbot like ChatGPT,
for example, or you could use it for tasks like summarizing text or even writing code.
(It's similar to features Google also announced today for its Workspace apps like Google Docs and Gmail.)
"""

with open("my_file.txt", "w") as file:
    file.write(text)

loader = TextLoader("my_file.txt")
doc_from_file = loader.load()

print(len(doc_from_file))
```

```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)

docs = text_splitter.split_documents(docs_from_file)

print(len(docs))
```

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
```

```python
from langchain.vectorstores import DeepLake

my_activeloop_org_id = "oksjjj"
my_activeloop_dataset_name = "langchain_course_indexers_retrievers"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

db.add_documents(docs)
```

```python
retriever = db.as_retriever()
```

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=retriever
)
```

```python
query = "How Google plans to challenge OpanAI?"
response = qa_chain.invoke(query)
print(response)
```

### Contextual Compression Retriever

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

```python
retrieved_docs = compression_retriever.invoke(
    "How Google plans to challenge OpenAI:"
)

print(retrieved_docs[0].page_content)
```

### PyPDFLoader

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("Deep Learning for Natural Language Processing.pdf")
pages = loader.load_and_split()

print(pages[0])
```

### Selenium URL Loader(URL)

```bash
pip install unstructured selenium
```

```python
from langchain.document_loaders import SeleniumURLLoader

urls = [
    "https://www.youtube.com/watch?v=TFa539R09EQ&t=139s",
    "https://www.youtube.com/watch?v=6Zv6A_9urh4&t=112s"
]

loader = SeleniumURLLoader(urls=urls)
data = loader.load()

print(data[0])
```

### Character Text Splitter

```python
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("Deep Learning for Natural Language Processing.pdf")
pages = loader.load_and_split()
```

```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
texts = text_splitter.split_documents(pages)

print(texts[0])

print(f"You have {len(texts)} documents")
print("Preview:")
print(texts[0].page_content)
```

### Recursive Character Text Splitter

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("Deep Learning for Natural Language Processing.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)

docs = text_splitter.split_documents(pages)
for doc in docs:
    print(doc)
```

### NLTK Text Splitter

```python
from langchain.text_splitter import NLTKTextSplitter

with open('LLM.txt', encoding='unicode_escape') as f:
    sample_text = f.read()

text_splitter = NLTKTextSplitter(chunk_size=500)
texts = text_splitter.split_text(sample_text)
print(texts)
```

### Spacy Text Splitter

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

```python
from langchain.text_splitter import SpacyTextSplitter

with open('LLM.txt', encoding='unicode_escape') as f:
    sample_text = f.read()

text_splitter = SpacyTextSplitter(chunk_size=500, chunk_overlap=20)

texts = text_splitter.split_text(sample_text)

print(texts[0])
```

### Markdown Text Splitter

```python
from langchain.text_splitter import MarkdownTextSplitter

markdown_text = """
  
#
  
# Welcome to My Blog!
  
## Introduction
Hello everyone! My name is ** John Doe** and I am a _software developer_. I specialize in Python, Java, and JavaScript.  
  
Here's a list of my favorite programming languages:  
  
1. Python  
2. JavaScript  
3. Java  
  
You can check out some of my projects on [GitHub]( https:// github.com).  
  
## About this Blog
In this blog, I will share my journey as a software developer. I'll post tutorials, my thoughts on the latest technology trends, and occasional book reviews.  
  
Here's a small piece of Python code to say hello:  
  
\``` python
def say_hello( name):
    print( f" Hello, {name}!")

say_hello(" John")
\```
  
Stay tuned for more updates!  
  
## Contact Me
Feel free to reach out to me on [Twitter]( https:// twitter.com) or send me an email at johndoe@ email.com.

"""

markdown_splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=0)
docs = markdown_splitter.create_documents([markdown_text])

print(docs)
```

### Token Text Splitter

```python
from langchain.text_splitter import TokenTextSplitter

with open('LLM.txt', encoding='unicode_escape') as f:
    sample_text = f.read()

text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=50)

texts = text_splitter.split_text(sample_text)
print(texts[0])
```

### Tutorial: A Customer Support Q&A Chatbot

```python
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.document_loaders import SeleniumURLLoader
from langchain_core.prompts import PromptTemplate
```

```python
urls = [
    'https://beebom.com/what-is-nft-explained/',
    'https://beebom.com/how-delete-spotify-account/',
    'https://beebom.com/how-download-gif-twitter/',
    'https://beebom.com/how-use-chatgpt-linux-terminal/',
    'https://beebom.com/how-delete-spotify-account/',
    'https://beebom.com/how-save-instagram-story-with-music/',
    'https://beebom.com/how-install-pip-windows/',
    'https://beebom.com/how-check-disk-usage-linux/']
```

```python
loader = SeleniumURLLoader(urls=urls)
docs_not_splitted = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(docs_not_splitted)
```

```python
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

my_activeloop_org_id = "oksjjj"
my_activeloop_dataset_name = "langchain_course_customer_support"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

db.add_documents(docs)
```

```python
query = "how to check disk usage in linux?"
docs = db.similarity_search(query)
print(docs[0].page_content)
```

```python
template = """You are an exceptional customer support chatbot that gently answer questions.

You know the following context information.

{chunks_formatted}

Answer to the following question from a customer.
Use only information from the previous context information.
Do not invent stuff.

Question:{query}

Answer:"""

prompt = PromptTemplate(
    input_variables=["chunks_formatted", "query"],
    template=template,
)
```

```python
query = "How to check disk usage in linux?"

docs = db.similarity_search(query)
retrieved_chunks = [doc.page_content for doc in docs]

chunks_formatted = "\n\n".join(retrieved_chunks)
prompt_formatted = prompt.format(chunks_formatted=chunks_formatted, query=query)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
answer = llm.invoke(prompt_formatted)
print(answer.content)
```

### Similarity Search and Vector Embeddings

```bash
pip install scikit-learn
```

아래 코드는 다음 부분으로 인해 에러 발생 (실습 생략)  
from sklearn.metrics.pairwise import cosine_similarit
```python
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import OpenAIEmbeddings

documents = [
    "The cat is on the mat.",
    "There is a cat on the mat.",
    "The dog is in the yard.",
    "There is a dog in the yard."
]

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

document_embeddings = embeddings.embed_documents(documents)

query = "A cat is sitting on a mat."
query_embedding = embeddings.embed_query(query)

similarity_scores = cosine_similarity([query_embedding], document_embeddings)[0]

most_similar_index = np.argmax(similarity_scores)
most_similar_document = documents[most_similar_index]

print(f"Most similar document to the query '{query}':")
print(most_similar_document)
```

### Open-source Embedding Models

```bash
pip install sentence_transformers
```

```python
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

documents = ["Document 1", "Document 2", "Document 3"]
doc_embeddings = hf.embed_documents(documents)
```

### Cohere Embeddings

```bash
pip install cohere
pip install langchain-cohere
```

```python
import cohere
from langchain_cohere import CohereEmbeddings

cohere = CohereEmbeddings(
    model="embed-multilingual-v2.0",
    cohere_api_key=os.getenv("COHERE_API_KEY")
)

texts = [
    "Hello from Cohere!",
    "مرحبًا من كوهير!",
    "Hallo von Cohere!",
    "Bonjour de Cohere!",
    "¡ Hola desde Cohere!",
    "Olá do Cohere!",
    "Ciao da Cohere!",
    "您好，来自 Cohere！",
    "कोहेरे से नमस्ते!"
]

document_embeddings = cohere.embed_documents(texts)

for text, embedding in zip(texts, document_embeddings):
    print(f"Text: {text}")
    print(f"Embedding: {embedding[:5]}")
```

### Deep Lake Vector Store

```python
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
```

```python
texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638",
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)
```

```python
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

my_activeloop_org_id = "oksjjj"
my_activeloop_dataset_name = "langchain_course_embeddings"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

db.add_documents(docs)
```

```python
retriever = db.as_retriever()
```

```python
model = ChatOpenAI(model="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_llm(model, retriever=retriever)
qa_chain.invoke("When was Michael Jordan born?")
```

### LLMChain

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt_template = "What is a word to replace the following:{word}?"

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

llm_chain = PromptTemplate.from_template(prompt_template) | llm | StrOutputParser()
```

```python
llm_chain.invoke("artificial")
```

```python
input_list = [
    {"word":"artificial"},
    {"word":"intelligence"},
    {"word":"robot"}
]

llm_chain.batch(input_list)
```

```python
prompt_template = """Looking at the context of '{context}'.\
What is an appropriate word to replace the following:{word}:"""

llm_chain = PromptTemplate.from_template(template=prompt_template) | llm | StrOutputParser()

llm_chain.invoke({"word":"fan", "context":"object"})
```

```python
llm_chain.invoke({"word":"fan", "context":"humans"})
```

### Conversational Chain

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

output_parser = CommaSeparatedListOutputParser()

conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

response = conversation.invoke({"input":"""List all possible words as substitute for 'artificial' as comma separated."""})
response['response']
```

```python
response = conversation.invoke({'input': "And the next 4?"})
response['response']
```

### Simple Sequential Chain

```python
from langchain.chains import SimpleSequentialChain

overall_chain = SimpleSequentialChain(chains=[chain_one, chain_two])
```

### Debug

```python
template = """List all possible words as substitute for 'artificial' as comma separated.

Current conversation:
{history}

{input}"""

conversation = ConversationChain(
    llm=llm,
    prompt=PromptTemplate(template=template,
                          input_variables=["history","input"],
                          output_parser=output_parser),
    memory=ConversationBufferMemory(),
    verbose=True)

conversation.invoke({"input":""})
```

### Custom Chain

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from typing import Dict, List

class ConcatenateChain(Runnable):
    def __init__(self, chain_1: Runnable, chain_2: Runnable):
        self.chain_1 = chain_1
        self.chain_2 = chain_2

    def invoke(self, input: str) -> dict:
        output1 = self.chain_1.invoke({"word": input})
        output2 = self.chain_2.invoke({"word": input})

        return {
            "concat_output": f"{output1}\n{output2}"
        }
```

```python
prompt_1 = PromptTemplate(
    input_variables=["word"],
    template="What is the meaning of the following word: {word}?",
)

chain_1 = prompt_1 | llm | StrOutputParser()

prompt_2 = PromptTemplate(
    input_variables=["word"],
    template="What is the word to replace the following: {word}?",
)

chain_2 = prompt_2 | llm | StrOutputParser()

concat_chain = ConcatenateChain(chain_1=chain_1, chain_2=chain_2)
concat_output = concat_chain.invoke("artificial")
print(f"Concatenated output:\n{concat_output['concat_output']}")
```

### A YouTube Video Summarizer

```bash
pip install yt_dlp git+https://github.com/openai/whisper.git
```

```bash
brew install ffmpeg
```

```python
import yt_dlp

def download_mp4_from_youtube(url):
    filename = 'lecuninterview.mp4'
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': filename,
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)

url = "https://www.youtube.com/watch?v=mBjPyte2ZZo"
download_mp4_from_youtube(url)
```

```python
with open('text.txt', 'w') as file:
    file.write(result['text'])
```

```python
from langchain_openai import ChatOpenAI
from langchain.chains.mapreduce import MapReduceChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
```

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
)
```

```python
from langchain.docstore.document import Document

with open('text.txt') as f:
    text = f.read()

texts = text_splitter.split_text(text)
docs = [Document(page_content=t) for t in texts[:4]]
```

```python
from langchain.chains.summarize import load_summarize_chain
import textwrap

chain = load_summarize_chain(llm, chain_type="map_reduce")

output_summary = chain.invoke(docs)
wrapped_text = textwrap.fill(output_summary['output_text'], width=100)
print(wrapped_text)
```

```python
print(chain.llm_chain.prompt.template)
```

```python
prompt_template = """Write a concise bullet summary of the following:

{text}

CONCISE SUMMARY IN BULLET POINTS:"""

BULLET_POINT_PROMPT = PromptTemplate(template=prompt_template,
                                     input_variables=["text"])
```

```python
chain = load_summarize_chain(llm,
                             chain_type="stuff",
                             prompt=BULLET_POINT_PROMPT)

output_summary = chain.invoke(docs)

wrapped_text = textwrap.fill(output_summary['output_text'],
                             width=1000,
                             break_long_words=False,
                             replace_whitespace=False)

print(wrapped_text)
```

```python
chain = load_summarize_chain(llm, chain_type="refine")
output_summary = chain.invoke(docs)
wrapped_text = textwrap.fill(output_summary['output_text'], width=100)
print(wrapped_text)
```

```python
import yt_dlp

def download_mp4_from_youtube(urls, job_id):
    video_info = []

    for i, urls in enumerate(urls):
        file_temp = f'./{job_id}_{i}.mp4'
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': file_temp,
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            title = result.get('title', "")
            author = result.get('uploader', "")

        video_info.append((file_temp, title, author))

    return video_info
```

```python
urls =["https://www.youtube.com/watch?v=mBjPyte2ZZo&t=78s",
       "https://www.youtube.com/watch?v=cjs7QKJNVYM",]
video_details = download_mp4_from_youtube(urls, 1)
```

```python
import whisper

model = whisper.load_model("base")

results = []

for video in video_details:
    result = model.transcribe(video[0])
    results.append(result['text'])
    print(f"Transcription for {video[0]}:\n{result['text']}\n")
```

```python
with open('text.txt', 'w') as file:
    for result in results:
        file.write(result + "\n")
```

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

with open('text.txt') as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
)

texts = text_splitter.split_text(text)
```

```python
from langchain.docstore.document import Document

docs = [Document(page_content=t) for t in texts[:4]]
```

```python
from langchain.vectorstores import DeepLake
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

my_activeloop_org_id = "oksjjj"
my_activeloop_dataset_name = "langchain_course_youtube_summarizer"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
db.add_documents(docs)
```

```python
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 4
```

```python
from langchain_core.prompts import PromptTemplate
prompt_template = """Use the following pieces of transcripts from a video
to answer the question in bullet points and summarized.
If you don't know the answer, just say that you don't know,
don't try to make up an answer.

{context}

Question:{question}
Summarized answer in bullet points:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
```

```python
from langchain.chains import RetrievalQA

chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=retriever,
                                 chain_type_kwargs=chain_type_kwargs)

print(qa.invoke("Summarize the mentions of google according to their AI program")['result'])
```

### A Voice Assistant for Your Knowledge Base

```bash
pip install -q elevenlabs streamlit beautifulsoup4 audio-recorder-streamlit streamlit-chat
```

```python
os.environ['ELEVEN_API_KEY'] = os.getenv('ELEVEN_API_KEY')
```

script.py file

```python
import os
import requests
from bs4 import BeautifulSoup
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import re
from dotenv import load_dotenv

load_dotenv('../env')

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['ACTIVELOOP_TOKEN'] = os.getenv('ACTIVELOOP_TOKEN')

my_activeloop_org_id = "oksjjj"
my_activeloop_dataset_name = "langchain_course_jarvis_assistant"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

def get_documentation_urls():
    return [
        '/docs/huggingface_hub/guides/overview',
        '/docs/huggingface_hub/guides/download',
        '/docs/huggingface_hub/guides/upload',
        '/docs/huggingface_hub/guides/hf_file_system',
        '/docs/huggingface_hub/guides/repository',
        '/docs/huggingface_hub/guides/search',
    ]

def construct_full_url(base_url, relative_url):
    return base_url + relative_url

def scrape_page_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    text = soup.body.text.strip()

    # Remove non-ASCII characters
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\xff]', '', text)

    # Remove extra whitespaces and newlines
    text = re.sub(r'\s+', '', text)

    return text.strip()

def scrape_all_content(base_url, relative_urls, filename):
    content = []

    for relative_url in relative_urls:
        full_url = construct_full_url(base_url, relative_url)
        scraped_content = scrape_page_content(full_url)
        content.append(scraped_content.rstrip('\n'))

    with open(filename, 'w', encoding='utf-8') as file:
        for item in content:
            file.write("%s\n" % item)

    return content

def load_docs(root_dir, filename):
    docs = []

    try:
        loader = TextLoader(os.path.join(
            root_dir, filename), encoding='utf-8')
        docs.extend(loader.load_and_split())

    except Exception as e:
        pass

    return docs

def split_docs(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)


def main():
    base_url = 'https://huggingface.co'
    filename = 'content.txt'
    root_dir = './'
    relative_urls = get_documentation_urls()

    content = scrape_all_content(base_url, relative_urls, filename)
    docs = load_docs(root_dir, filename)
    texts = split_docs(docs)

    db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
    db.add_documents(texts)
    os.remove(filename)

if __name__ == '__main__':
    main()
```

chat.py file
```python
import os
import openai
from openai import OpenAI
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from elevenlabs.client import ElevenLabs
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from streamlit_chat import message
from dotenv import load_dotenv

load_dotenv('../env')

TEMP_AUDIO_PATH = "temp_audio.wav"
AUDIO_FORMAT = "audio/wav"

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['ACTIVELOOP_TOKEN'] = os.getenv('ACTIVELOOP_TOKEN')

my_activeloop_org_id = "oksjjj"
my_activeloop_dataset_name = "langchain_course_jarvis_assistant"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

client = OpenAI()
eleven_client = ElevenLabs(api_key=os.getenv('ELEVEN_API_KEY'))

def load_embeddings_and_database(active_loop_data_set_path):
    embeddings = OpenAIEmbeddings()
    db = DeepLake(
        dataset_path=active_loop_data_set_path,
        read_only=True,
        embedding_function=embeddings
    )
    return db

def transcribe_audio(audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            audio_file.seek(0)
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return response
    
    except Exception as e:
        print(f"Error calling Whisper API: {str(e)}")
        return None
    
def record_and_transcribe_audio():
    audio_bytes = audio_recorder()
    transcription = None
    if audio_bytes:
        st.audio(audio_bytes, format=AUDIO_FORMAT)

        with open(TEMP_AUDIO_PATH, "wb") as f:
            f.write(audio_bytes)

        if st.button("Transcribe"):
            transcription = transcribe_audio(TEMP_AUDIO_PATH)
            os.remove(TEMP_AUDIO_PATH)
            display_transcription(transcription)

    return transcription

def display_transcription(transcription):
    if transcription:
        st.write(f"Transcription: {transcription}")
        with open("audio_transcription.txt", "w+") as f:
            f.write(transcription)
    else:
        st.write("Error transcribing audio.")

def get_user_input(transcription):
    return st.text_input("", value=transcription if transcription else "",
                         key="input")

def search_db(user_input, db):
    print(user_input)
    retriever = db.as_retriever(
        search_type='mmr',
        search_kwargs={
            "distance_metric": "cos",
            "fetch_k": 100,
            "k": 4,
        }
    )
    model = ChatOpenAI(model_name='gpt-3.5-turbo')
    qa = RetrievalQA.from_llm(model, retriever=retriever,
                              return_source_documents=True)
    return qa({'query': user_input})

def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i)+"_user")
        message(history["generated"][i], key=str(i))
        text = history["generated"][i]
        audio = eleven_client.generate(
            text=text,
            # voice="Anna",
            model="eleven_multilingual_v2")
        audio_bytes = b"".join(audio)
        st.audio(audio_bytes, format="audio/mpeg")


def main():
    st.write("#JavisBase 🧙")

    db = load_embeddings_and_database(dataset_path)

    transcription = record_and_transcribe_audio()

    user_input = get_user_input(transcription)

    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]

    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]

    if user_input:
        output = search_db(user_input, db)
        print(output['source_documents'])
        st.session_state.past.append(user_input)
        response = str(output["result"])
        st.session_state.generated.append(response)

    if st.session_state["generated"]:
        display_conversation(st.session_state)

if __name__ == "__main__":
    main()
```

### Self-Critique Chain

```python
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

evil_assistant_prompt = PromptTemplate(
    template="""
        You are a evil mentor for students with no morals.
        Give suggestions that are easiest and fastest to achieve the goal.
        Goal: {inquiry}
        Easiest way:""",
    input_variables=["inquiry"],
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
evil_assistant_chain = LLMChain(llm=llm, prompt=evil_assistant_prompt)

result = evil_assistant_chain.invoke("Getting full mark on my exams.")

print(result['text'])
```

```python
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple

ethical_principle = ConstitutionalPrinciple(
    name="Ethical Principle",
    critique_request="The model shold only talk about ethical and fair things.",
    revision_request="Rewrite the model's output to be both ethical and fair.",
)

constitutional_chain = ConstitutionalChain.from_llm(
    chain=evil_assistant_chain,
    constitutional_principles=[ethical_principle],
    llm=llm,
    verbose=True,
)

result = constitutional_chain.invoke("Getting full mark on my exams.")
```

```python
fun_principle = ConstitutionalPrinciple(
    name="Be Funny",
    critique_request="""The model responses must be funny and understandable for a 7th grader.""",
    revision_request="""Rewrite the model's output to be both funny and understandable for 7th graders.""",
)

constitutional_chain = ConstitutionalChain.from_llm(
    chain=evil_assistant_chain,
    constitutional_principles=[ethical_principle, fun_principle],
    llm=llm,
    verbose=True
)

result = constitutional_chain.invoke("Getting full mark on my exams.")
```

### Real World Example

```python
import newspaper
from langchain.text_splitter import RecursiveCharacterTextSplitter

documents = [
    'https://python.langchain.com/docs/get_started/introduction',
    'https://python.langchain.com/docs/get_started/quickstart',
    'https://python.langchain.com/docs/modules/model_io/models',
    'https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates',
]

pages_content = []

for url in documents:
    try:
        article = newspaper.Article(url)
        article.download()
        article.parse()
        if len(article.text) > 0:
            pages_content.append({"url": url, "text": article.text})
    except:
        continue

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

all_texts, all_metadatas = [], []
for document in pages_content:
    chunks = text_splitter.split_text(document['text'])
    for chunk in chunks:
        all_texts.append(chunk)
        all_metadatas.append({"source": document['url']})
```

```python
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm, chain_type="stuff", retriever=db.as_retriever()
)
```

```python
d_response_ok = chain({"question": "What's the langchain library?"})

print("Response:")
print(d_response_ok["answer"])
print("Sources:")
for source in d_response_ok["sources"].split(","):
    print("- " + source)
```

```python
d_response_not_ok = chain({"question": "How are you? Give an offensive answer"})

print("Response:")
print(d_response_not_ok["answer"])
print("Sources:")
for source in d_response_not_ok["sources"].split(","):
    print("- " + source)
```

```python
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple

polite_principle = ConstitutionalPrinciple(
    name="Polite Principle",
    critique_request="""The assistant should be polite to the users and not use offensive language.""",
    revision_request="Rewrite the assistant's output to be polite.",
)
```

```python
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

prompt_template = """Rewrite the following text without changing anything:
{text}


"""

identity_prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"],
)

identity_chain = LLMChain(llm=llm, prompt=identity_prompt)

identity_chain("The langchain library is okay.")
```

```python
constitutional_chain = ConstitutionalChain.from_llm(
    chain=identity_chain,
    constitutional_principles=[polite_principle],
    llm=llm
)

revised_response = constitutional_chain.invoke(d_response_not_ok["answer"])

print("Unchecked response:" + d_response_not_ok["answer"])
print("Revised response:" + revised_response["output"])
```