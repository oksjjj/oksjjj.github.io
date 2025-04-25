---
title: Ch#7 - BUILDING LLMS FOR PRODUCTION 코드 현행화
author: oksjjj
date: 2025-04-14 19:00:00 +0900
categories: [LLM/RAG]
tags: [LLM, RAG, langchain, chat, gpt, llama, index, deeplake]
render_with_liquid: false
image:
  path: /assets/img/thumbnail/building_llms_code.png
  alt: BUILDING LLMS FOR PRODUCTION 코드 현행화
  src: "https://oksjjj.github.io/building_llms_code.png"
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