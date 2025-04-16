---
title: Ch#10 - Elasticsearch The DEFINITIVE GUIDE 쿼리 현행화
author: oksjjj
date: 2025-04-16 21:20:00 +0900
categories: [Elasticsearch - The DEFINITIVE GUIDE]
tags: [Elasticsearch]
render_with_liquid: false
image:
  path: /assets/img/thumbnail/elasticsearch_correct_query.png
  alt: Elasticsearch The DEFINITIVE GUIDE 쿼리 현행화
  src: "https://oksjjj.github.io/elasticsearch_correct_query.png"
---

### Index Settings

인덱스 샐성
```
PUT /my_temp_index
{
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    }
}
```

인덱스 수정
```
PUT /my_temp_index/_settings
{
    "number_of_replicas": 1
}
```

인덱스 삭제
```
DELETE /my_temp_index
```

### Configuring Analyzers

스페인어 불용어 Analyzer
```
PUT /spanish_docs
{
    "settings": {
        "analysis": {
            "analyzer": {
                "es_std": {
                    "type": "standard",
                    "stopwords": "_spanish_"
                }
            }
        }
    }
}
```

스페인 불용어 테스트 (El 제거)
```
POST /spanish_docs/_analyze
{
    "analyzer": "es_std",
    "text": "El veloz zorro marrón"
}
```

인덱스 삭제
```
DELETE /spanish_docs
```

### Custom Analysis

Custom Analyzer 만들기
```
PUT /my_index
{
    "settings": {
        "analysis": {
            "char_filter": {
                "&_to_and": {
                    "type": "mapping",
                    "mappings": [ "&=> and " ]
                }
            },
            "filter": {
                "my_stopwords": {
                    "type": "stop",
                    "stopwords": [ "the", "a" ]
                }
            },
            "analyzer": {
                "my_analyzer": {
                    "type": "custom",
                    "char_filter": [ "html_strip", "&_to_and" ],
                    "tokenizer": "standard",
                    "filter": [ "lowercase", "my_stopwords" ]
                }
            }
        }
    }
}
```

테스트
```
POST /my_index/_analyze
{
    "analyzer": "my_analyzer",
    "text": "The quick & brown fox"
}
```

필드에 대해 analyzer 지정
```
PUT /my_index/_mapping
{
    "properties": {
        "title": {
            "type": "text",
            "analyzer": "my_analyzer"
        }
    }
}
```