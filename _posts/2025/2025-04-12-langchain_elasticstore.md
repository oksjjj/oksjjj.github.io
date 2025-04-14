---
title: LangChain - Elasticstore
author: oksjjj
date: 2025-04-12 14:20:00 +0900
categories: [LLM/RAG]
tags: [llamaindex, elasticsearch]
render_with_liquid: false
image:
  path: /assets/img/thumbnail/rag_code.png
  alt: RAG Driven generative AI 실습 코드 현행화
  src: "https://oksjjj.github.io/rag_code.png"
---

### python 버전

3.9

### 데이터 생성

```bash
pip install beautifulsoup4 requests markdown sentence-transformers

```

```python
import requests
from bs4 import BeautifulSoup
import re

# URLs of the Wikipedia articles
urls = [
    "https://en.wikipedia.org/wiki/Space_exploration",
    "https://en.wikipedia.org/wiki/Apollo_program",
    "https://en.wikipedia.org/wiki/Hubble_Space_Telescope",
    "https://en.wikipedia.org/wiki/Mars_rover",  # Corrected link
    "https://en.wikipedia.org/wiki/International_Space_Station",
    "https://en.wikipedia.org/wiki/SpaceX",
    "https://en.wikipedia.org/wiki/Juno_(spacecraft)",
    "https://en.wikipedia.org/wiki/Voyager_program",
    "https://en.wikipedia.org/wiki/Galileo_(spacecraft)",
    "https://en.wikipedia.org/wiki/Kepler_Space_Telescope",
    "https://en.wikipedia.org/wiki/James_Webb_Space_Telescope",
    "https://en.wikipedia.org/wiki/Space_Shuttle",
    "https://en.wikipedia.org/wiki/Artemis_program",
    "https://en.wikipedia.org/wiki/Skylab",
    "https://en.wikipedia.org/wiki/NASA",
    "https://en.wikipedia.org/wiki/European_Space_Agency",
    "https://en.wikipedia.org/wiki/Ariane_(rocket_family)",
    "https://en.wikipedia.org/wiki/Spitzer_Space_Telescope",
    "https://en.wikipedia.org/wiki/New_Horizons",
    "https://en.wikipedia.org/wiki/Cassini%E2%80%93Huygens",
    "https://en.wikipedia.org/wiki/Curiosity_(rover)",
    "https://en.wikipedia.org/wiki/Perseverance_(rover)",
    "https://en.wikipedia.org/wiki/InSight",
    "https://en.wikipedia.org/wiki/OSIRIS-REx",
    "https://en.wikipedia.org/wiki/Parker_Solar_Probe",
    "https://en.wikipedia.org/wiki/BepiColombo",
    "https://en.wikipedia.org/wiki/Juice_(spacecraft)",
    "https://en.wikipedia.org/wiki/Solar_Orbiter",
    "https://en.wikipedia.org/wiki/CHEOPS_(satellite)",
    "https://en.wikipedia.org/wiki/Gaia_(spacecraft)"
]

def clean_text(content):
    content = re.sub(r'\[\d+\]', '', content)
    return content

def fetch_and_clean(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.find('div', {'class': 'mw-parser-output'})

    for section_title in ['References', 'Bibliography', 'External links', 'See also']:
        section = content.find('span', id=section_title)

        if section:
            for sib in section.parent.find_next_siblings():
                sib.decompose()
            section.parent.decompose()
    text = content.get_text(separator=' ', strip=True)
    text = clean_text(text)
    return text

with open("llm.txt", "w", encoding='utf-8') as file:
    for url in urls:
        clean_article_text = fetch_and_clean(url)
        file.write(clean_article_text + '\n')
print("Content written to llm.txt")
```

### Elasticstore 만들기

```bash
pip install langchain-elasticsearch elasticsearch langchain_openai
```

```python
from dotenv import load_dotenv
load_dotenv("../env")

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

source_text = "llm.txt"

with open(source_text, "r") as f:
    text = f.read()
CHUNK_SIZE = 1000
chunked_text = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
```

```python
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings
from elasticsearch import Elasticsearch


es = Elasticsearch(
    "https://127.0.0.1:9200",
    basic_auth=('elastic', 'password'),
    verify_certs=False)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = ElasticsearchStore(
    index_name="space_exploration_v1",
    embedding=embeddings,
    es_connection=es
)

metadata_list = [{"source": source_text} for _ in chunked_text]
ids = vector_store.add_texts(
    texts=chunked_text,
    metadatas=metadata_list
)

print(f"성공적으로 {len(ids)}개의 문서가 저장되었습니다.")
```

### Elasticstore 이용하기

```python
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings
from elasticsearch import Elasticsearch


es = Elasticsearch(
    "https://127.0.0.1:9200",
    basic_auth=('elastic', 'password'),
    verify_certs=False)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = ElasticsearchStore(
    index_name="space_exploration_v1",
    embedding=embeddings,
    es_connection=es
)
```

```python
user_prompt = "Tell me about space exploration on the Moon and Mars."

results = vector_store.similarity_search_with_score(query=user_prompt, k=1)
```

```python
def wrap_text(text, width=80):
    lines = []
    while len(text) > width:
        split_index = text.rfind(' ', 0, width)
        if split_index == -1:
            split_index = width
        lines.append(text[:split_index])
        text = text[split_index:].strip()
    lines.append(text)
    return '\n'.join(lines)
```

```python
top_doc, top_score = results[0]
top_text = top_doc.page_content.strip()
top_metadata = top_doc.metadata.get('source', 'N/A')

print("Top Search Result:")
print(f"Score: {top_score:.4f}")
print(f"Source: {top_metadata}")
print("Text:")
print(wrap_text(top_text))
```