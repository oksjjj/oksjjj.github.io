---
title: Ch#7 - BUILDING LLMS FOR PRODUCTION 코드 현행화
author: oksjjj
date: 2025-05-16 13:00:00 +0900
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

### Query Engine

```bash
mkdir -p './paul_graham/'
pip install -q faiss-cpu llama-index-vector-stores-faiss
```

[paul_graham_essay.txt](https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt)  
  
Deep Lake로는 계속 에러가 발생해서, FaissVectorStore를 이용해서 구현  
  
```python
import numpy as np
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

# 1. 문서 로딩
documents = SimpleDirectoryReader('./paul_graham').load_data()

# 2. 파싱 및 청크
Settings.chunk_size = 512
Settings.chunk_overlap = 64

node_parser = SimpleNodeParser.from_defaults(
    chunk_size=Settings.chunk_size,
    chunk_overlap=Settings.chunk_overlap,
)
nodes = node_parser.get_nodes_from_documents(documents)

embed_model = OpenAIEmbedding()

for node in nodes:
    emb = embed_model.get_text_embedding(node.text)
    # embedding을 list(float)로 보장
    if isinstance(emb, np.ndarray):
        node.embedding = emb.tolist()
    else:
        node.embedding = list(map(float, emb))
```
  
```python
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore

# dimension은 임베딩 벡터 차원 수 (예: 1536)
dimension = 1536

# 빈 FAISS 인덱스 생성 (L2 거리 기준)
faiss_index = faiss.IndexFlatL2(dimension)

# FaissVectorStore에 넘기기
vector_store = FaissVectorStore(faiss_index)
```
  
```python
from llama_index.core import StorageContext, VectorStoreIndex

storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
```

```python
query_engine = vector_index.as_query_engine(streaming=True, similarity_top_k=10)
```

```python
streaming_response = query_engine.query(
    "What does Paul Graham do?",
)
streaming_response.print_response_stream()
```

### Sub Question Query Engine
  
```python
query_engine = vector_index.as_query_engine(similarity_top_k=10)
```

```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine

query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="pg_essay",
            description="Paul Graham essay on What I Worked On",
        ),
    ),
]

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=True,
)
```

```python
response = query_engine.query(
    "How was Paul Graham's life different before, during, and after YC?"
)
print(">>>The final response:\n", response)
```

### Reranking

```python
import cohere
import os

os.environ['COHERE_API_KEY'] = os.getenv('COHERE_API_KEY')

co = cohere.Client(os.environ['COHERE_API_KEY'])

query = "What is the capital of the United States?"

documents = [
    """ Carson City is the capital city of the American state of Nevada. At the 2010 United States Census, Carson City had a population of 55,274.""",
    """ The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.""",
    """ Charlotte Amalie is the capital and largest city of the United States Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.""",
    """ Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district. """,
    """ Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.""",
    """ North Dakota is a state in the United States. 672,591 people lived in North Dakota in the year 2010. The capital and seat of government is Bismarck."""
]
```

```python
results = co.rerank(query=query, documents=documents, top_n=3,
                    model='rerank-english-v3.0')
for idx, r in enumerate(results.results):
    print(f"Document Rank: {idx+1}, Documents INdex: {r.index}")
    print(f"Document: {documents[r.index]}")
    print(f"Relevance Score: {r.relevance_score:.2f}")
    print("\n")
```

```bash
pip install -q llama-index-postprocessor-cohere-rerank
```

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank

cohere_rerank = CohereRerank(api_key=os.environ['COHERE_API_KEY'], top_n=2)
```

```python
query_engine = vector_index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[cohere_rerank],
)

response = query_engine.query(
    "What did Sam Altman do in this essay?",
)
print(response)
```