---
layout: post
title: "[빅데이터와 정보검색] 4주차 ElasticSearch"
date: 2025-09-27 08:00:00 +0900
categories:
  - "대학원 수업"
  - "빅데이터와 정보검색"
tags: []
---

> 출처: 빅데이터와 정보검색 – 황영숙 교수님, 고려대학교 (2025)

## p2. ElasticSearch  

- **Elastic 창시**: 2012년, 4인의 멤버가 회사로 설립  
  - 2004년 **사이 배논(Shay Bannon)**  
    - 요리 레시피 검색엔진을 개발 후 **아파치 루씬(Apache Lucene)** 을 적용하려던 중 루씬이 가진 한계를 발견  
    - 루씬의 한계를 보완하기 위해 새로운 검색엔진 프로젝트로 **ElasticSearch** 출발  
  - **창시 멤버**: 사이 배논, **스티븐 서르만(Steven Schuurman)**, **우리 보네스(Uri Boness)**, **사이먼 윌너(Simon Willnauer)** 4인의 멤버가 회사로 설립  

- **Elasticsearch**는 **자바(Java)** 로 만들어진 **루씬 기반의 오픈 소스 검색 엔진**  
  - 뛰어난 검색능력과 **대규모 분산 시스템**을 구축할 수 있는 다양한 기능들을 제공  

---

## p3. ElasticSearch  

- **Elasticsearch**는 **자바(Java)** 로 만들어진 **루씬 기반의 오픈 소스 검색 엔진**  

[특징]  
- **오픈소스 (open source)** ([GitHub 링크](https://github.com/elastic)), Apache 2.0 라이선스로 배포  
- **실시간 분석 (real-time)**: 실시간에 가까운 속도로 색인된 데이터의 검색, 집계가 가능  
- **전문(full text) 검색 엔진**  
  - 루씬은 기본적으로 **역파일 색인(inverted file index)** 라는 구조로 데이터를 저장  
  - Elasticsearch도 루씬 기반 위에서 개발되어 색인된 모든 데이터를 역파일 색인 구조로 저장  
  - **JSON** 형식으로 데이터를 전달, 내부적으로는 역파일 색인 구조로 저장  
  - Elasticsearch에서 **질의(query)** 에 사용되는 쿼리문이나 결과도 모두 **JSON** 형식으로 전달되고 리턴  

- **RESTFul API**  
  - **마이크로 서비스 아키텍처(MSA)** 를 기본으로 설계  
  - Rest API를 기반으로 지원하며 모든 데이터 조회, 입력, 삭제를 **HTTP** 프로토콜을 통해 처리  

- **멀티테넌시(multitenancy)**  
  - 데이터들은 **인덱스(Index)** 라는 논리적 집합 단위로 구성되며, 서로 다른 저장소에 분산되어 저장  
  - 서로 다른 인덱스들을 별도의 커넥션 없이 하나의 질의로 묶어서 검색하고, 결과들을 하나의 출력으로 도출  

---

## p4. ElasticSearch의 논리적 구조  

| **Elasticsearch 개념** | **RDBMS 대응**       | **설명** |
|----------------------|-------------------|-------------------------------------------------------------------|
| **Cluster**          | Database Server 전체 | 여러 인덱스를 포함하는 하나의 검색/저장 시스템 단위 |
| **Index**            | Database          | 문서를 저장하는 논리적 공간 (특정 주제별, 서비스별로 나눔) |
| **Type** (deprecated)| Table             | 하나의 인덱스 안에 있는 여러 문서를 묶어서 타임이라는 논리 단위.<br> 현재는 사용되지 않음 |
| **Document**         | Row (Record)      | **JSON 형식의 데이터 레코드** |
| **Field**            | Column            | 문서(Document)의 속성(키-값) |
| **_id**              | Primary Key       | 인덱스 내 문서에 부여되는 고유한 구분자.<br> 인덱스 이름과 _id 조합은 Elasticsearch 클러스터 내에서 고유함 |
| **Mapping**          | Schema            | 필드의 자료형 정의 (text, keyword, date, number 등) |

---

## p5. ElasticSearch의 물리적 구조  

| Elasticsearch 개념 | 설명 |
|---------------------|------|
| **Node**           | Elasticsearch가 실행되는 단일 서버/프로세스. 클러스터에 여러 노드 참여 가능. |
| **Cluster**        | 하나 이상의 노드가 모여 이루는 전체 시스템. 클러스터 이름으로 구분. |
| **Shard**          | 인덱스를 나누어 저장하는 단위(분산 저장). 검색·색인을 병렬 처리 가능. |
| **Primary Shard**  | 원본 데이터를 저장하는 샤드. |
| **Replica Shard**  | Primary Shard의 복제본. 장애 대비/로드 밸런싱에 사용. |
| **Gateway**        | 클러스터가 재시작될 때 인덱스와 샤드를 복구하는 저장소 역할. |

<img src="/assets/img/bigdatasearch/4/image_1.png" alt="image" width="720px">  

---

## p6. Elasticsearch RestAPIs  

| 엘라스틱서치에서 사용하는 HTTP 메서드 | 기능              | 데이터베이스 질의 문법 |
|----------------------------------|-----------------|------------------|
| **GET**                          | 데이터 조회        | SELECT           |
| **PUT**                          | 데이터 생성        | INSERT           |
| **POST**                         | 인덱스 업데이트, 조회 | UPDATE, SELECT   |
| **DELETE**                       | 데이터 삭제        | DELETE           |
| **HEAD**                         | 인덱스의 정보 확인   |                  |

---

## p7. ELK stack  

1. **Logstash/Beats**  
- 역할: Ingest(수집)  
- 다양한 소스(DB, CSV, Log 등)으로부터 데이터를 가져다, 조작 후 **Elasticsearch**에게 전달하는 역할  

2. **Elasticsearch**  
- 역할: Store, Search, Analyze  
- 직접 수집한 데이터 또는 Logstash으로 수집한 데이터를 관리, 조작(검색)  

3. **Kibana**  
- 역할: Visualize(시각화), Manage(관리), Bulk 데이터 입력  
- **Elasticsearch**의 데이터를 시각화하거나 편리하게 조작 기능을 제공  
- GUI 환경을 제공한다  

---

## p8. ELK stack  

<img src="/assets/img/bigdatasearch/4/image_2.png" alt="image" width="720px">  