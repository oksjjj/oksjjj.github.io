---
title: Elasticsearch - The DEFINITIVE GUIDE - Chapter#1
author: oksjjj
date: 2025-03-25 22:18:00 +0900
categories: [Elasticsearch - The DEFINITIVE GUIDE]
tags: [Elasticsearch]
render_with_liquid: false
image:
  path: /assets/img/thumbnail/elasticsearch_correct_query.png
  alt: (Book Review) Elasticsearch - The DEFINITIVE GUIDE 쿼리 현행화
  src: "https://oksjjj.github.io/elasticsearch_correct_query.png"
---


### 클러스터 안의 문서 개수 검색하기
```
GET /_count
{
    "query": {
        "match_all": {}
    }
}
```
### Employee#1 입력하기
```
PUT /megacorp/_doc/1
{
    "first_name": "John",
    "last_name": "Smith",
    "age": 25,
    "about": "I love to go rock climbing",
    "interests": [ "sports", "music" ]
}
```
### Employee#2 입력하기
```
PUT /megacorp/_doc/2
{
    "first_name": "Jane",
    "last_name": "Smith",
    "age": 32,
    "about": "I Like to collect rock albums",
    "interests": [ "music" ]
}
```
### Employee#3 입력하기
```
PUT /megacorp/_doc/3
{
    "first_name": "Douglas",
    "last_name": "Fir",
    "age": 35,
    "about": "I Like to build cabinets",
    "interests": [ "forestry" ]
}
```
### 문서 조회하기 - Employee#1
```
GET /megacorp/_doc/1
```
### 모든 employee 검색하기
```
GET /megacorp/_search
```
### 성이 Smith인 employee 검색하기
```
GET /megacorp/_search?q=last_name:Smith
```
### query DSL(domain-specific language)로 검색하기
```
GET /megacorp/_search
{
    "query": {
        "match": {
            "last_name": "Smith"
        }
    }
}
```
### 나이가 30보다 많고, 성이 Smith인 employee 검색하기
```
GET /megacorp/_search
{
  "query": {
    "bool": {
      "must": {
        "match": {
          "last_name": "smith"
        }
      },
      "filter": {
        "range": {
          "age": { "gt": 30 }
        }
      }
    }
  }
}
```
### Full-Text Search
```
GET /megacorp/_search
{
    "query": {
        "match": {
            "about": "rock climbing"
        }
    }
}
```
### Phrase Search
```
GET /megacorp/_search
{
    "query": {
        "match_phrase": {
            "about": "rock climbing"
        }
    }
}
```
### 검색 사유 하이라이트 하기
```
GET /megacorp/_search
{
    "query": {
        "match_phrase": {
            "about": "rock climbing"
        }
    },
    "highlight": {
        "fields": {
            "about": {}
        }
    }
}
```
### Aggregation
```
GET /megacorp/_search
{
  "aggs": {
    "all_interests": {
      "terms": { "field": "interests.keyword" }
    }
  }
}
```
### 성이 Smith인 사람에 한해 Aggregation
```
GET /megacorp/_search
{
    "query": {
        "match": {
            "last_name": "smith"
        }
    },
    "aggs": {
        "all_interests": {
            "terms": {
                "field": "interests.keyword"
            }
        }
    }
}
```
### interests별 평균 연령 구하기 (Aggregation 중첩)
```
GET /megacorp/_search
{
    "aggs": {
        "all_interests": {
            "terms": { "field": "interests.keyword" },
            "aggs": {
                "avg_age": {
                    "avg": { "field": "age" }
                }
            }
        }
    }
}
```