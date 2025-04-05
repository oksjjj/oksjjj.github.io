---
title: Elasticsearch - The DEFINITIVE GUIDE - Chapter#8
author: oksjjj
date: 2025-04-03 21:30:00 +0900
categories: [Elasticsearch - The DEFINITIVE GUIDE]
tags: [Elasticsearch]
render_with_liquid: false
image:
  path: /assets/img/thumbnail/elasticsearch_correct_query.png
  alt: (Book Review) Elasticsearch - The DEFINITIVE GUIDE 쿼리 현행화
  src: "https://oksjjj.github.io/elasticsearch_correct_query.png"
---

### 샘플 데이터 넣기
```
POST /tweets/_bulk
{ "index": { "_id": "1" } }
{ "tweet": "How to manage text effectively", "user_id": 2, "date": "2025-04-03T10:00:00" }
{ "index": { "_id": "2" } }
{ "tweet": "Learn to manage text search with Elasticsearch", "user_id": 2, "date": "2025-04-02T15:30:00" }
{ "index": { "_id": "3" } }
{ "tweet": "Exploring advanced text search techniques", "user_id": 3, "date": "2025-04-01T08:45:00" }
{ "index": { "_id": "4" } }
{ "tweet": "Learn to manage text search with Elasticsearch", "user_id": 2, "date": "2025-04-03T10:00:00" }
{ "index": { "_id": "5" } }
{ "tweet": "Learn to manage text search with Elasticsearch", "user_id": 1, "date": "2025-04-01T08:45:00" }
{ "index": { "_id": "6" } }
{ "tweet": "Learn to manage text search with Elasticsearch", "user_id": 1, "date": "2025-04-02T15:30:00" }
```
  
### Sorting
user_id가 1인 결과들은 모두 _score가 의미없음
```
GET /_search
{
    "query": {
        "bool": {
            "filter": { "term": { "user_id": 1 }}
        }
    }
}
```
  
### Sorting by Field Values
date로 정렬하기 때문에 max_score, _score는 null이고, sort는 날짜값
```
GET /_search
{
    "query": {
        "bool": {
            "filter": { "term": { "user_id": 1 }}
        }
    },
    "sort": { "date": { "order": "desc" }}
}
```
  
### Multilevel Sorting  
쿼리 후 2가지 조건으로 정렬하기
```
GET /_search
{
    "query": {
        "bool": {
            "must": { "match": { "tweet": "manage text search" }},
            "filter": { "term": { "user_id": 2 }}
        }
    },
    "sort": [
        { "date": { "order": "desc" }},
        { "_score": { "order": "desc" }}
    ]
}
```
  
인덱스 정리
```
DELETE /tweets
```

### String Sorting and Multifields
인덱스 만들기
```
PUT /gb
{
  "mappings": {
    "properties": {
      "tweet": {
        "type": "text",
        "analyzer": "english",
        "fields": {
          "raw": {
            "type": "keyword"
          }
        }
      }
    }
  }
}
```
  
샘플 데이터 넣기
```
POST /gb/_bulk
{ "index": {} }
{ "tweet": "I love to go rock climbing" }
{ "index": {} }
{ "tweet": "I like to collect rock albums" }
{ "index": {} }
{ "tweet": "I like to build cabinets" }
```
  
검색은 full text search로 정렬은 not_analyzed로
```
GET /_search
{
    "query": {
        "match": { "tweet": "rock climbing" }
    },
    "sort": "tweet.raw"
}
```
  
인덱스 정리
```
DELETE /gb
```