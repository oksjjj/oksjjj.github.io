---
title: Ch#8 - Elasticsearch The DEFINITIVE GUIDE 쿼리 현행화
author: oksjjj
date: 2025-04-03 21:30:00 +0900
categories: [Elasticsearch - The DEFINITIVE GUIDE]
tags: [Elasticsearch]
render_with_liquid: false
---

### Sorting
샘플 데이터 넣기
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

### Understanding the Score
샘플 데이터 넣기
```
POST /us/_bulk
{ "index": { "_index": "us", "_id": "1" } }
{ "user_id": 1, "tweet": "Our honeymoon in Bali was unforgettable. Best honeymoon ever!" }
{ "index": { "_index": "us", "_id": "2" } }
{ "user_id": 2, "tweet": "Just got back from an amazing vacation in Hawaii." }
{ "index": { "_index": "us", "_id": "3" } }
{ "user_id": 3, "tweet": "Planning my dream trip to Paris next month." }
{ "index": { "_index": "us", "_id": "4" } }
{ "user_id": 4, "tweet": "Honeymoon planning can be stressful, but it's worth it." }
{ "index": { "_index": "us", "_id": "5" } }
{ "user_id": 5, "tweet": "The weather is perfect for a beach day!" }
{ "index": { "_index": "us", "_id": "6" } }
{ "user_id": 6, "tweet": "Looking for recommendations for romantic getaways." }
{ "index": { "_index": "us", "_id": "7" } }
{ "user_id": 7, "tweet": "Our wedding was beautiful, now excited for the honeymoon!" }
{ "index": { "_index": "us", "_id": "8" } }
{ "user_id": 8, "tweet": "Just booked a cruise for our anniversary celebration." }
{ "index": { "_index": "us", "_id": "9" } }
{ "user_id": 9, "tweet": "Trying to decide between a mountain retreat or a tropical paradise." }
{ "index": { "_index": "us", "_id": "10" } }
{ "user_id": 10, "tweet": "Honeymoon destinations: Maldives, Santorini, or Bora Bora? So many honeymoon options!" }
```  
  
_score 산정 로직을 설명하게 하기  
```
GET /_search?explain=true
{
    "query": { "match": { "tweet": "honeymoon" }}
}
```
  
yaml 파일로 출력하기
```
GET /_search?explain=true&format=yaml
{
    "query": { "match": { "tweet": "honeymoon" }}
}
```
  
Understanding Why a Document Matched
```
GET /us/_explain/4
{
    "query": {
        "bool": {
            "filter": { "term": { "user_id": 2 }},
            "must": { "match": { "tweet": "honeymoon" }}
        }
    }
}
```