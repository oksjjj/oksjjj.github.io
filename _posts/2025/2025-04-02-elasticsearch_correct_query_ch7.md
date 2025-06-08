---
title: Ch#7 - Elasticsearch The DEFINITIVE GUIDE 쿼리 현행화
author: oksjjj
date: 2025-04-02 18:45:00 +0900
categories: [Elasticsearch - The DEFINITIVE GUIDE]
tags: [Elasticsearch]
render_with_liquid: false
---


### 샘플 데이터 넣기
```
POST /emails/_bulk
{ "index": { "_id": 1 } }
{ "email": "business opportunity", "folder": "inbox" }
{ "index": { "_id": 2 } }
{ "email": "urgent business proposal", "folder": "inbox" }
{ "index": { "_id": 3 } }
{ "email": "casual conversation", "folder": "sent" }
{ "index": { "_id": 4 } }
{ "email": "team meeting notes", "folder": "inbox" }
```
  

### Filtering a Query
```
GET /_search
{
    "query": {
        "bool": {
            "must": { "match": { "email": "business opportunity" }},
            "filter": { "term": { "folder": "inbox" }}
        }
    }
}
```
  
### Just a Query
```
GET /_search
{
    "query": {
        "bool": {
            "filter": { "term": { "folder": "inbox" }}
        }
    }
}
```
  
### Just a Query (위와 동일한 쿼리)
```
GET /_search
{
    "query": {
        "bool": {
            "must": { "match_all": {}},
            "filter": { "term": { "folder": "inbox" }}
        }
    }
}
```
  
### A Query as a Filter
```
GET /_search
{
    "query": {
        "bool": {
            "filter": { "term": { "folder": "inbox" }},
            "must_not": { "match": { "email": "urgent business proposal" }}
        }
    }
}
```
  
인덱스 정리
```
DELETE /emails
```
  
### Validating Queries
```
GET /gb/_validate/query
{
    "query": {
        "tweet": { "match": "really powerful" }
    }
}
```
  
### Understanding Errors
```
GET /gb/_validate/query?explain
{
    "query": {
        "tweet": { "match": "really powerful" }
    }
}
```
  
### Understanding Queries
```
GET /_validate/query?explain
{
    "query": {
        "match": { "tweet": "really powerful" }
    }
}
```