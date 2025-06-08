---
title: Ch#6 - Elasticsearch The DEFINITIVE GUIDE 쿼리 현행화
author: oksjjj
date: 2025-03-29 07:45:00 +0900
categories: [Elasticsearch - The DEFINITIVE GUIDE]
tags: [Elasticsearch]
render_with_liquid: false
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