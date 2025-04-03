---
title: Elasticsearch - The DEFINITIVE GUIDE - Chapter#6
author: oksjjj
date: 2025-03-29 07:45:00 +0900
categories: [Elasticsearch - The DEFINITIVE GUIDE]
tags: [Elasticsearch]
render_with_liquid: false
image:
  path: /assets/img/thumbnail/elasticsearch_correct_query.png
  alt: (Book Review) Elasticsearch - The DEFINITIVE GUIDE 쿼리 현행화
  src: "https://oksjjj.github.io/elasticsearch_correct_query.png"
---


### Testing Analyzers
```
POST /_analyze
{
  "analyzer": "standard",
  "text": "Text to analyze"
}
```
  
  
### 분석되는 필드 생성 (type: text)
```
PUT /gb
{
    "mappings": {
        "properties": {
            "tweet": {
                "type": "text",
                "analyzer": "english"
            },
            "date": {
                "type": "date"
            },
            "name": {
                "type": "text"
            },
            "user_id": {
                "type": "long"
            }
        }
    }
}
```


### 분석되지 않는 필드 추가 (type: keyword)
```
PUT /gb/_mapping
{
    "properties": {
        "tag": {
            "type": "keyword"
        }
    }
}
```


### 분석되는 필드 테스트
```
POST /gb/_analyze
{
    "field": "tweet",
    "text": "Black-cats"
}
```


### 분석되지 않는는 필드 테스트
```
POST /gb/_analyze
{
    "field": "tag",
    "text": "Black-cats"
}
```
  
인덱스 정리
```
DELETE /gb
```