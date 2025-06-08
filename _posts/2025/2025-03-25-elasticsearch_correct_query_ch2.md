---
title: Ch#2 - Elasticsearch The DEFINITIVE GUIDE 쿼리 현행화
author: oksjjj
date: 2025-03-25 22:25:00 +0900
categories: [Elasticsearch - The DEFINITIVE GUIDE]
tags: [Elasticsearch]
render_with_liquid: false
---


### Cluster Health
```
GET /_cluster/health
```
### 인덱스 생성하기 (샤드 3개, 레플리카 1개)
```
PUT /blogs
{
    "settings": {
        "number_of_shards": 3,
        "number_of_replicas": 1
    }
}
```
### Cluster Health 재확인
```
GET /_cluster/health
```
### 레플리카 개수 변경
```
PUT /blogs/_settings
{
    "number_of_replicas": 2
}
```
### Cluster Health 재확인
```
GET /_cluster/health
```
  
인덱스 정리
```
DELETE /blogs
```