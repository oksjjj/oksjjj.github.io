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
from langchain import PromptTemplate
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