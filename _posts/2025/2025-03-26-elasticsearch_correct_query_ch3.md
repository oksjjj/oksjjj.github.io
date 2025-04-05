---
title: Elasticsearch - The DEFINITIVE GUIDE - Chapter#3
author: oksjjj
date: 2025-03-26 20:48:00 +0900
categories: [Elasticsearch - The DEFINITIVE GUIDE]
tags: [Elasticsearch]
render_with_liquid: false
image:
  path: /assets/img/thumbnail/elasticsearch_correct_query.png
  alt: (Book Review) Elasticsearch - The DEFINITIVE GUIDE 쿼리 현행화
  src: "https://oksjjj.github.io/elasticsearch_correct_query.png"
---


### id를 123으로 지정해서 생성 (PUT)
```
PUT /website/_doc/123
{
    "title": "My first blog entry",
    "text": "Just trying this out...",
    "date": "2014/01/01"
}
```
### id를 지정하지 않고 생성 (POST)
```
POST /website/_doc/
{
    "title": "My second blog entry",
    "text": "Still trying this out...",
    "date": "2014/01/01"
}
```
### id가 123인 문서 조회
```
GET /website/_doc/123
```
### id가 124인 문서 조회
```
GET /website/_doc/124
```
### id가 123인 문서에 대해서 title, text만 조회
```
GET /website/_doc/123?_source=title,text
```
### id가 123인 문서에 대해서 _source 조회
```
GET /website/_doc/123?_source
```
### id가 123인 문서가 존재하는지 확인
```
HEAD /website/_doc/123
```
### id가 124인 문서가 존재하는지 확인
```
HEAD /website/_doc/124
```
### id가 123인 문서 업데이트 (_version이 2가 되는지 확인)
```
PUT /website/_doc/123
{
    "title": "My first blog entry",
    "text": "I am starting to get the hang of this...",
    "date": "2014/01/02"
}
```
### id가 123인 문서 삭제
```
DELETE /website/_doc/123
```
### id가 1인 문서 생성
```
PUT /website/_doc/1?op_type=create
{
    "title": "My first blog entry",
    "text": "Just trying this out..."
}
```
### id가 1인 문서 조회
```
GET /website/_doc/1
```
### seq_no가 4일 경우에만 id가 1인 문서 업데이트 (충돌 방지)
```
PUT /website/_doc/1?if_seq_no=4&if_primary_term=1
{
    "title": "My first blog entry",
    "text": "Starting to get the hang of this..."
}
```
### id가 2이고, version_no가 5이며, version_type이 external인 문서 생성
```
PUT /website/_doc/2?version=5&version_type=external
{
    "title": "My first external blog entry",
    "text": "Starting to get the hang of this..."
}
```
### version_no를 10으로 업데이트
```
PUT /website/_doc/2?version=10&version_type=external
{
    "title": "My first external blog entry",
    "text": "This is a piece of cake..."
}
```
### id가 1인 문서 부분 업데이트
```
POST /website/_update/1
{
    "doc": {
        "tags": [ "testing" ],
        "views": 0
    }
}
```
### id가 1인 문서 확인
```
GET /website/_doc/1
```
### script를 이용하여 views 1 증가
```
POST /website/_update/1
{
    "script": "ctx._source.views+=1"
}
```
### id가 1인 문서 확인
```
GET /website/_doc/1
```
### script를 이용하여 tag 추가
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
### id가 1인 문서 확인
```
GET /website/_doc/1
```
### script를 이용하여 views가 1일 경우 삭제하기
```
POST /website/_update/1
{
    "script": {
        "source": "ctx.op = ctx._source.views == params.count ? 'delete' : 'none'",
        "params": {
            "count": 1
        }
    }
}
```
### id가 1인 문서 확인
```
GET /website/_doc/1
```
### views 라는 항목이 없을 경우 1로 초기화하기 (upsert)
```
POST /website/_update/1
{
  "script": "ctx._source.views+=1",
  "upsert": {
    "views": 1
  }
}
```
### id가 1인 문서 확인
```
GET /website/_doc/1
```
### 충돌이 발생할 경우 재시도 카운트 지정하기
```
POST /website/_update/1?retry_on_conflict=5
{
  "script": "ctx._source.views+=1",
  "upsert": {
    "views": 0
  }
}
```
### id가 1인 문서 확인
```
GET /website/_doc/1
```
### mget으로 여러 문서 동시에 조회하기
```
GET /_mget
{
    "docs": [
        {
            "_index": "website",
            "_id": 2
        },
        {
            "_index": "website",
            "_id": 1,
            "_source": "views"
        }
    ]
}
```
### mget 단축 형태
```
GET /website/_mget
{
    "ids": ["2", "1"]
}
```
### bulk (첫번째 명령이 실패하더라도 나머지 라인은 진행)
```
POST /_bulk
{ "delete": { "_index": "website", "_id": "123"}}
{ "create": { "_index": "website", "_id": "123"}}
{ "title": "My first blog post"}
{ "index": { "_index": "website" }}
{ "title": "My second blog post" }
{ "update": { "_index": "website", "_id": "123", "retry_on_conflict": 3}}
{ "doc": { "title": "My updated blog post" }}
```
  
인덱스 정리
```
DELETE /website
```