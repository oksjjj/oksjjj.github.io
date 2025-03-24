---
title: (Book Review) Elasticsearch - The DEFINITIVE GUIDE 쿼리 현행화
author: oksjjj
date: 2025-03-24 21:00:00 +0900
categories: [Book Review]
tags: [Elasticsearch]
render_with_liquid: false
image:
  path: /assets/img/thumbnail/elasticsearch_correct_query.png
  alt: (Book Review) Elasticsearch - The DEFINITIVE GUIDE 쿼리 현행화
  src: "https://oksjjj.github.io/elasticsearch_correct_query.png"
---

## **사유**

이 책은 Elasticsearch 1.X 버전에 기반하여 작성되었으며,  
현재 Elasticsearch는 8.X 버전이어서 문법의 많은 변화가 있었다.  
그래서 문법 변경에 맞게 쿼리를 현행화하였다.  


## **쿼리 현행화**

### **Chapter#1**
  
#### 수정#1
**Before**
```
PUT /megacorp/employee/1
{
  "first_name": "John",
  "last_name": "Smith",
  "age": 25,
  "about": "I love to go rock climbing",
  "interests": [ "sports", "music" ]
}
```  
**After**  
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
  
#### 수정#2
**Before**
```
GET /megacorp/employee/1
```  
**After**  
```
GET /megacorp/_doc/1
```  
  
#### 수정#3
**Before**
```
GET /megacorp/employee/_search
```  
**After**  
```
GET /megacorp/_doc/_search
```  
  
#### 수정#4
**Before**
```
GET /megacorp/employee/_search?q=last_name:Smith
```  
**After**  
```
GET /megacorp/_search?q=last_name:Smith
```  
  
#### 수정#5
**Before**
```
GET /megacorp/employee/_search
{
  "query": {
    "match": {
      "last_name": "Smith"
    }
  }
}
```  
**After**  
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
  
#### 수정#6
**Before**
```
GET /megacorp/employee/_search
{
  "query": {
    "filtered": {
      "filter": {
        "range": {
          "age": { "gt" : 30 }
        }
      },
      "query": {
        "match": {
          "last_name": "smith"
        }
      }
    }
  }
}
```  
**After**  
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

#### 수정#7
**Before**
```
GET /megacorp/employee/_search
{
  "aggs": {
    "all_interests": {
      "terms": { "field": "interests" }
    }
  }
}
```  
**After**  
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
  
### **Chapter#3**
  
#### 수정#1
**Before**
```
PUT /website/blog/1/_create
{
  "title": "My first blog entry",
  "text": "Just trying this out..."
}
```  
**After**  
```
PUT /website/_doc/1/?op_type=create
{
  "title": "My first blog entry",
  "text": "Just trying this out..."
}
```  
  
#### 수정#2
**Before**
```
PUT /website/blog/1?version=1
{
  "title": "My first blog entry",
  "text": "starting to get the hang of this..."
}
```  
**After**  
```
PUT /website/_doc/1?if_seq_no=4&if_primary_term=1
{
  "title": "My first blog entry",
  "text": "starting to get the hang of this..."
}
```  
  
#### 수정#3
**Before**
```
PUT /website/blog/2?version=5&version_type=external
{
  "title": "My first external blog entry",
  "text": "starting to get the hang of this..."
}
```  
**After**  
```
PUT /website/_doc/2?version=5&version_type=external
{
  "title": "My first external blog entry",
  "text": "starting to get the hang of this..."
}
```  
  
#### 수정#4
**Before**
```
POST /website/blog/1/_update
{
  "doc": {
    "tags": [ "testing" ],
    "views": 0
  }
}
```  
**After**  
```
POST /website/_update/1
{
  "doc": {
    "tags": [ "testing" ],
    "views": 0
  }
}
```  

#### 수정#5
**Before**
```
POST /website/blog/1/_update
{
  "script": "ctx._source.tags+=new_tag",
  "params": {
    "new_tag": "search"
  }
}
```  
**After**  
```
POST /website/_update/1
{
  "script": {
    "source": "ctx._source.tags.add(params.new_tag)",
    "params": {
      "new_tag": "search"
    }
  }
}
```  