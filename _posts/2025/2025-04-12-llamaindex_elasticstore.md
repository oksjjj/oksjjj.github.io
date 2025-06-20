---
title: Llama Index - Elasticstore
author: oksjjj
date: 2025-04-12 14:30:00 +0900
categories: [LLM/RAG]
tags: [llamaindex, elasticsearch]
render_with_liquid: false
---

### python 버전

3.9

### 실습 코드
  
```bash
pip install openai==1.70.0 llama-index==0.12 beautifulsoup4 requests
pip install elasticsearch llama-index-vector-stores-elasticsearch
pip install llama_index.embeddings.huggingface
```

```python
from dotenv import load_dotenv
load_dotenv('../env')
```

```python
import openai
import os
import pandas as pd

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
```

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
```

```python
import requests
from bs4 import BeautifulSoup
import re
import os

urls = [
    "https://github.com/VisDrone/VisDrone-Dataset",
    "https://paperswithcode.com/dataset/visdrone",
    "https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Zhu_VisDrone-DET2018_The_Vision_Meets_Drone_Object_Detection_in_Image_Challenge_ECCVW_2018_paper.pdf",
    "https://github.com/VisDrone/VisDrone2018-MOT-toolkit",
    "https://en.wikipedia.org/wiki/Object_detection",
    "https://en.wikipedia.org/wiki/Computer_vision",
    "https://en.wikipedia.org/wiki/Convolutional_neural_network",
    "https://en.wikipedia.org/wiki/Unmanned_aerial_vehicle",
    "https://www.faa.gov/uas",
    "https://www.tensorflow.org",
    "https://pytorch.org",
    "https://keras.io",
    "https://arxiv.org/abs/1804.06985",
    "https://arxiv.org/abs/2202.11983",
    "https://motchallenge.net",
    "http://www.cvlibs.net/datasets/kitti",
    "https://www.dronedeploy.com",
    "https://www.dji.com",
    "https://arxiv.org",
    "https://openaccess.thecvf.com",
    "https://roboflow.com",
    "https://www.kaggle.com",
    "https://paperswithcode.com",
    "https://github.com"
]
```

```python
def clean_text(content):
    content = re.sub(r'\[\d+\]', '', content)   # Remove references
    content = re.sub(r'[^\w\s.]', '', content)  # Remove punctuation (except periods)
    return content

def fetch_and_clean(url):
    try:
        response = requests.get(url)
        response.raise_for_status()     # Raise exception for bad responses (e.g., 404)
        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.find('div', {'class': 'mw-parser-output'}) or soup.find('div', {'id': 'content'})

        if content is None:
            return None
        
        for section_title in ['References', 'Bibliograph', 'External links', 'See also', 'Notes']:
            section = content.find('span', id=section_title)
            while section:
                for sib in section.parent.find_next_siblings():
                    sib.decompose()
                section.parent.decompose()
                section = content.find('span', id=section_title)

        text = content.get_text(separator=' ', strip=True)
        text = clean_text(text)
        return text
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return None
```

```python
output_dir = './data/'
os.makedirs(output_dir, exist_ok=True)

for url in urls:
    article_name = url.split('/')[-1].replace('.html','')
    filename = os.path.join(output_dir, article_name + '.txt')
    clean_article_text = fetch_and_clean(url)
    if clean_article_text:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(clean_article_text)

print(f"Content(ones that were possible) written to files in the '{output_dir} directory.")
```

```python
documents = SimpleDirectoryReader("./data/").load_data()
documents[0]
```

```python
from elasticsearch import AsyncElasticsearch


# Elasticsearch 연결 및 LlamaIndex 저장소 구성
es_client = AsyncElasticsearch(
    "https://127.0.0.1:9200",
    basic_auth=('elastic', 'password'),
    verify_certs=False)

es_index_name = "drone_knowledge_base"
```

```python
vector_store = ElasticsearchStore(
    index_name=es_index_name,
    es_client=es_client,
    vector_dim=384,  # SentenceTransformer: all-MiniLM-L6-V2 기준
    recreate_index=True
)
```

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

```python
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context)
```

```python
k = 3
temp = 0.1
mt = 1024
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-V2')

def calculate_cosine_similarity_with_embeddings(text1, text2):
    embeddings1 = model.encode(text1)
    embeddings2 = model.encode(text2)
    similarity = cosine_similarity([embeddings1], [embeddings2])
    return similarity[0][0]
```

```python
query_engine = index.as_query_engine(similarity_top_k=k, temperature=temp, num_output=mt)
```

```python
import textwrap
import pandas as pd

# 사용자 쿼리 처리
def index_query(input_query):
    response = query_engine.query(input_query)
    print("\n" + textwrap.fill(str(response), 100) + "\n")
    node_data = []
    for node_with_score in response.source_nodes:
        node = node_with_score.node
        node_data.append({
            'Node_ID': node.id_,
            'Score': node_with_score.score,
            'Text': node.text
        })
    return pd.DataFrame(node_data), response
```

```python
import time

# 실행 예시
user_input = "How are drones used in wars?"

start_time = time.time()
df, response = index_query(user_input)
end_time = time.time()

print(f"Query execution time: {end_time - start_time:.4f} seconds\n")
print(df.to_markdown(index=False))
```

```python
for node_with_score in response.source_nodes:
    node = node_with_score.node
    print(f"Node ID: {node.id_}, Chunk Size: {len(node.text)} characters")
```