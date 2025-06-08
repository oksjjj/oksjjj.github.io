---
title: Ch#5 - BUILDING LLMS FOR PRODUCTION 코드 현행화
author: oksjjj
date: 2025-04-12 15:20:00 +0900
categories: [LLM/RAG]
tags: [LLM, RAG, langchain, chat, gpt, llama, index, deeplake]
render_with_liquid: false
---

## python 버전

3.9

```python
from dotenv import load_dotenv
import os
load_dotenv('../env')

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['ACTIVELOOP_TOKEN'] = os.getenv('ACTIVELOOP_TOKEN')
```
  
### Prompt Templates

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = "You are an assistant that helps users find information about movies."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "Find information about the movie {movie_title}."
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                               human_message_prompt])

response = chat.invoke(chat_prompt.format_prompt(movie_title="Inception").to_messages())

print(response.content)
```

### Summarization Chain Example

```bash
pip install pypdf
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

summarize_chain = load_summarize_chain(llm)

document_loader = PyPDFLoader(file_path="The One Page Linux Manual.pdf")
document = document_loader.load()

summary = summarize_chain.invoke(document)
print(summary['output_text'])
```

### Q&A Chain Example

```python
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(template="Question:{question}\nAnswer:",
                        input_variables=["question"])

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
chain = prompt | llm | StrOutputParser()

chain.invoke("what is the meaning of life?")
```

### Building a News Articles Summarizer

```bash
pip install newspaper3k
pip install "lxml[html_clean]"
```

```python
import requests
from newspaper import Article

headers = {
    'User-Agent': """Mozilla/5.0(Windows NT 10.0;Win64;x64) AppleWebKit/537.76(KHTML,like Gecko) Chrome/89.0.4389.82 Safari/537.36"""
}

article_url = """https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-super-computer-will-set-records/"""

session = requests.Session()

try:
    response = session.get(article_url, headers=headers, timeout=10)

    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()

        print(f"Title:{article.title}")
        print(f"Text:{article.text}")

    else:
        print(f"Failed to fetch article at {article_url}")

except Exception as e:
    print(f"Error occurred while fetching article at {article_url}: {e}")
```

```python
from langchain.schema import HumanMessage

template = """You are a very good assistant that summarizes online articles.

Here's the article you want to summarize.

=============================
Title:{article_title}

{article_text}
=============================

Write a summary of the previous article.
"""

prompt = template.format(article_title=article.title, article_text=article.text)

messages = [HumanMessage(content=prompt)]
```

```python
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
```

```python
summary = chat.invoke(messages)
print(summary.content)
```

```python
template = """You are an advanced AI assistant that summarizes online articles into bulleted lists.

Here's the article you need to summarize.

=============================
Title:{article_title}

{article_text}
=============================

Now, provide a summarized version of the article in a bulleted list format.
"""

prompt = template.format(article_title=article.title, article_text=article.text)

summary = chat.invoke([HumanMessage(content=prompt)])
print(summary.content)
```

```python
template = """You are an advanced AI assistant that summarizes online articles into bulleted lists.

Here's the article you need to summarize.

=============================
Title:{article_title}

{article_text}
=============================

Now, provide a summarized version of the article in a bulleted list format, in French.
"""

prompt = template.format(article_title=article.title, article_text=article.text)

summary = chat.invoke([HumanMessage(content=prompt)])
print(summary.content)
```

## LlamaIndex

### Data Connectors

```bash
pip install llama-index
pip install wikipedia
```

```python
from typing import List
import wikipedia
from llama_index.core.schema import Document
from llama_index.core.readers.base import BaseReader

class WikipediaReader(BaseReader):
    """Reader for loading Wikipedia pages."""

    def load_data(self, pages: List[str]) -> List[Document]:
        documents = []
        for page in pages:
            try:
                content = wikipedia.page(page).content
                documents.append(Document(text=content))
            except Exception as e:
                print(f"Error loading page '{page}': {e}")
        return documents

loader = WikipediaReader()
documents = loader.load_data(["Natural language processing"])
print(len(documents))
```

### Nodes

```python
from llama_index.core.node_parser import SimpleNodeParser

parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=20)

nodes = parser.get_nodes_from_documents(documents)
print(len(nodes))
```

### Vector Store Index

```python
pip install llama_index.vector_stores.deeplake
pip install deeplake
pip install llama-index-embeddings-langchain
```

```python
from llama_index.vector_stores.deeplake import DeepLakeVectorStore

my_activeloop_org_id = "oksjjj"
my_activeloop_dataset_name = "LlamaIndex_intro"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=False)
```

```python
from llama_index.core import StorageContext, VectorStoreIndex

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
```

### Query Engines

```python
from llama_index.core import GPTVectorStoreIndex

index = GPTVectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What does NLP stands for?")
print(response.response)
```

### Saving and Loading Indexes Locally

```python
# store index as vector embeddings on the disk
index.storage_context.persist()
# This saves the data in the 'storage' by default
# to minimize repetitive processing
```

storage 디렉토리 삭제 후 실습  

```python
# Index Storage Checks
import os.path
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

from typing import List
import wikipedia
from llama_index.core.schema import Document
from llama_index.core.readers.base import BaseReader

class WikipediaReader(BaseReader):
    """Reader for loading Wikipedia pages."""

    def load_data(self, pages: List[str]) -> List[Document]:
        documents = []
        for page in pages:
            try:
                content = wikipedia.page(page).content
                documents.append(Document(text=content))
            except Exception as e:
                print(f"Error loading page '{page}': {e}")
        return documents

# Let's see if our index already exists in storage.
if not os.path.exists("./storage"):
    # If not, we'll load the Wikipedia data and create a new index
    loader = WikipediaReader()
    documents = loader.load_data(["Natural language processing", "Artificial Intelligence"])
    index = VectorStoreIndex.from_documents(documents)
    # Index storing
    index.storage_context.persist()

else:
    # If the index already exists, we'll just load it
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
```