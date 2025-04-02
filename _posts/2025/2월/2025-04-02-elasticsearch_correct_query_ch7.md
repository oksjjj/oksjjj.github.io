---
title: Elasticsearch - The DEFINITIVE GUIDE - Chapter#7
author: oksjjj
date: 2025-04-02 18:45:00 +0900
categories: [Elasticsearch - The DEFINITIVE GUIDE]
tags: [Elasticsearch]
render_with_liquid: false
image:
  path: /assets/img/thumbnail/elasticsearch_correct_query.png
  alt: (Book Review) Elasticsearch - The DEFINITIVE GUIDE 쿼리 현행화
  src: "https://oksjjj.github.io/elasticsearch_correct_query.png"
---


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