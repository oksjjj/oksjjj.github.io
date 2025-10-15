---
layout: post
title: "[빅데이터와 정보검색] 7주차 생성형AI 검색 기초 - 벡터검색과 RAG"
date: 2025-10-14 18:00:00 +0900
categories:
  - "대학원 수업"
  - "빅데이터와 정보검색"
tags: []
---

> 출처: 빅데이터와 정보검색 – 황영숙 교수님, 고려대학교 (2025)

## p2. 생성형 AI 검색의 정의와 특징  

- **생성형 AI 검색 (Generative AI Search)**  
  - 대규모 언어 모델(LLM)을 활용하여 사용자의 질문을 이해하고,  
    관련 정보를 검색한 후 이를 종합하여 자연스러운 맞춤형 답변을 생성하는 검색 기술  

- **전통적 검색 vs. 생성형 AI 검색**

| 구분 | 전통적 검색 | 생성형 AI 검색 |
|:--|:--|:--|
| **결과 형태** | 링크 목록 | 통합된 자연어 답변 |
| **정보 처리** | 키워드 매칭 | 의미 이해 및 종합 |
| **사용자 경험** | 여러 페이지 방문 필요 | 즉각적인 답변 제공 |
| **상호작용** | 단방향 검색 | 대화형 탐색 |
| **맥락 이해** | 제한적 | 깊은 맥락 파악 |

---

## p3. 생성형 AI 검색의 정의와 특징  

- **생성형 AI 검색의 특징**

  - **자연어 이해 및 생성**  
    - 복잡한 질문처리  
    - 맥락기반 해석: 이전 대화 맥락을 고려한 답변 생성  
    - 자연스러운 표현: 검색결과를 사람이 말하듯 자연스럽게 구성  

  - **정보 통합 및 종합**  
    - 다중 출처 결합: 여러 웹페이지, 문서의 정보를 하나의 답변으로 통합  
    - 요약 및 정리: 방대한 정보 중 핵심만 추출하여 제공  
    - 비교분석: 서로 다른 관점이나 데이터를 비교하여 제시  

  - **대화형 인터페이스**  
    - 연속적 질문: 추가 질문을 통해 정보를 점진적으로 탐색  
    - 질문정제: 모호한 질문을 명확히 하거나 추가정보 요청  
    - 맞춤형 응답: 사용자의 전문성 수준에 맞춘 설명  

  - **실시간 정보 접근**  
    - 최신 정보 검색: 웹크롤링을 통한 실시간 데이터 확보  
    - 동적 업데이트: 빠르게 변하는 정보 반영  
    - 시점인식: 질문의 시간적 맥락 이해  

---

## p4. 생성형 AI 검색의 정의와 특징  

- **주요 기술 요소**

  - **LLM (Large Language Model)** : 생성형 AI 검색의 핵심 기술  
    - 질문의 의도 파악  
    - 검색 쿼리 최적화  
    - 정보 종합 및 답변 생성  
    - 자연스러운 문장 구성  

  - **검색 증강 생성 (RAG: Retrieval Augmented Generation)**  
    - 외부 지식을 검색하여 LLM의 답변을 보강하는 기술  
    - **프로세스**  
      - 사용자 질문 분석  
      - 관련 문서/정보 검색  
      - 검색된 정보를 컨텍스트로 제공  
      - LLM이 컨텍스트 기반 답변 생성  

  - **의미기반 검색 (Semantic Search)**  
    - 단순 키워드 매칭이 아닌 의미기반 검색 수행  
    - **기술 요소**  
      - 벡터 임베딩  
      - 유사도 계산: 질문과 문서 간 의미적 거리 측정  
      - 밀집 검색(Dense Retrieval)을 통한 정확도 향상  

---

## p5. 검색증강생성 (RAG: Retrieval Augmented Generation)  

- **LLM의 근본적 한계**

  - **지식의 시점 고정 (Knowledge Cutoff)**  
    - LLM은 학습 시점까지의 데이터만 알고 있음  
    - 실시간 정보, 최신 뉴스, 업데이트된 데이터 접근 불가  

  - **환각 현상 (Hallucination)**  
    - 학습되지 않은 정보에 대해 그럴듯하지만 거짓인 답변을 생성  

  - **도메인 특화 지식 부족**  
    - 일반적 학습 데이터로는 특정 기업, 산업, 기술의 전문 지식이 부족  

  - **출처 검증 불가**  
    - 생성된 답변의 근거를 확인할 방법이 없음  

---

## p6. 검색증강생성 (RAG: Retrieval Augmented Generation)  

- **외부 지식 활용 → Fine Tuning 대비 시간과 비용 감소**  
  - 대규모의 구조화된 지식 베이스(예: 위키피디아)를 모델에 연결  
  - 주어진 질의에 대한 관련 정보를 지식 베이스에서 검색 및 추출  

- **증거 기반 답변 생성으로 Hallucination 완화**  
  - 검색된 지식 정보를 증거로 활용하여 보다 사실에 기반한 답변 생성  
  - 생성된 답변의 출처를 명시함으로써 신뢰성 향상  

- **맥락 이해력 향상**  
  - 외부 지식을 통해 질의의 배경 지식과 맥락 정보를 파악  
  - 단순한 패턴 매칭이 아닌 추론 능력을 바탕으로 한 답변 생성  

➡ **LLM 성능 향상 및 신뢰성 확보**

---

## p7. LLM 에이전트

<img src="/assets/img/lecture/bigdatasearch/7/image_1.png" alt="image" width="800px">  

---

## p8. RAG (Retrieval Augmented Generation)

- **RAG의 정의와 원리**  
  - **RAG = Retrieval (검색) + Augmented (증강) + Generation (생성)**  
  - 외부 지식 베이스에서 관련 정보를 실시간으로 검색하여,  
    이를 LLM의 컨텍스트에 추가함으로써 더 정확하고 최신의 답변을 생성하는 기술  

- **RAG vs Fine-tuning vs Prompt Engineering**

| 방식 | 장점 | 단점 | 적합한 경우 |
|:--|:--|:--|:--|
| **RAG** | • 실시간 업데이트<br>• 비용 효율적<br>• 출처 제공 | • 검색 품질 의존<br>• 지연 시간 | 최신 정보, 대용량 문서 |
| **Fine-tuning** | • 특정 작업 최적화<br>• 빠른 응답 | • 비용 高<br>• 업데이트 어려움 | 특정 스타일/형식 학습 |
| **Prompt Eng.** | • 즉시 적용<br>• 비용 無 | • 컨텍스트 제한<br>• 일관성 낮음 | 간단한 작업 |

---

## p9. RAG의 아키텍처 – RAG의 핵심 구성요소  

- **Retriever** : 외부 데이터(문서, DB 등)에서 관련 정보 검색  
  - 인덱싱 최적화: 청크 크기, 메타데이터, 혼합 검색  
  - 쿼리 최적화: 쿼리 확장, 변환, 재작성  
  - 검색 알고리즘: BM25, Dense Retriever, 하이브리드 등  

- **Augmentation** : 검색된 정보를 컨텍스트로 활용해 LLM 입력 확장  
  - 컨텍스트 재정렬, 요약, 정보 압축  
  - 중복 제거, 중요도 판단, 스타일/톤 일관성 유지  

- **Generator** : LLM이 검색 정보와 내재 지식을 결합해 응답 생성  
  - LLM의 프롬프트 설계  
  - 다중 문서 통합, 대화 이력 반영  
  - 파인튜닝 / 컨텍스트 학습  

---

## p10. RAG 시스템 아키텍처  

<img src="/assets/img/lecture/bigdatasearch/7/image_2.png" alt="image" width="800px">  

---

### 보충 설명

#### 1. **RAG 시스템의 전체 흐름**  
- RAG는 두 단계로 구성된다: **Indexing 단계**와 **Retrieval & Generation 단계**.  
- 먼저 문서들을 작은 단위(Chunk)로 나누고, 이를 벡터 형태로 변환하여 **Vector Store**에 저장한다.  
- 이후 사용자의 질의(Query)가 입력되면, 동일한 임베딩 공간으로 변환하여  
  벡터 유사도를 기반으로 관련 문서를 검색하고, 이 정보를 **LLM의 입력 컨텍스트로 통합**한다.  

#### 2. **Indexing 단계**  
- **문서 분할(Splitting/Chunking)** : 긴 문서를 의미 단위로 분할하여 효율적 검색이 가능하도록 한다.  
- **임베딩(Embedding)** : 각 청크를 고차원 벡터로 변환하여 의미적 정보를 수치화한다.  
- **저장(Indexing)** : 변환된 벡터를 **Vector Store**(예: FAISS, Pinecone 등)에 저장하여 검색 준비를 완료한다.  

#### 3. **Retrieval & Generation 단계**  
- **Query Processor** : 사용자의 질문과 프롬프트를 결합해 검색 가능한 질의로 변환한다.  
- **Retriever** : 벡터 유사도를 기준으로 관련 문서를 검색하여 **Top-K 문서**를 반환한다.  
- **Augmentation** : 검색된 문서를 요약·정제하여 **LLM 입력 컨텍스트**로 결합한다.  
- **Generative Model** : LLM이 입력된 컨텍스트와 내부 지식을 함께 활용해  
  문맥적·사실적인 응답을 생성한다.  

#### 4. **결과 생성 및 반환**  
- LLM이 생성한 응답은 검색된 문서의 정보를 반영하므로,  
  단순 생성 모델보다 **사실성(Factuality)** 과 **신뢰성(Reliability)** 이 높다.  
- 이 과정을 통해 RAG는 **지식의 최신성**과 **정확도**를 동시에 확보한다.  

---

## p11. RAG

- RAG 시스템 아키텍처  

<img src="/assets/img/lecture/bigdatasearch/7/image_3.png" alt="image" width="800px">  

1. Query가 입력되면 QueryEncoder를 통과하여 representation(q(x)) 생성  
2. q(x)와 가장 가까운(InnerProduct기준) {5,10}개의 Passage 탐색  
3. Passage를 기존 Query와 concat하여 Generator Input으로 사용  
4. 각 Passage별 생성결과를 Marginalize하여 최종 결과물 도출  

---

### 보충 설명  

#### 1. **RAG의 개요와 핵심 구조**  
- RAG(Retrieval-Augmented Generation)는 **검색 기반 생성 확률모델**로,  
  **Retriever $p_\eta(z|x)$** 와 **Generator $p_\theta(y|x,z)$** 를 결합한 형태이다.  
- 전체 모델은 다음과 같은 확률 형태로 표현된다.  

  $$
  p(y|x) = \sum_{z} p_\eta(z|x) \, p_\theta(y|x,z)
  $$  

  - $x$: 입력 질의(Query)  
  - $z$: 검색된 문서(Passage)  
  - $y$: 생성된 답변(Response)  
- 즉, RAG는 질의에 대해 관련 문서 $z$를 확률적으로 검색한 후,  
  각 문서를 기반으로 생성된 답변의 확률을 모두 **결합(Marginalization)** 하여  
  최종 응답을 생성한다.  

#### 2. **그림 속 구성요소 설명**  

- **Query Encoder**  
  - 입력된 질의 예시:  
    - *“Define ‘middle ear’”*  
    - *“Barack Obama was born in Hawaii.”*  
    - *“The Divine Comedy”*  
  - 각 질의는 **Query Encoder**를 통해 벡터 표현 $q(x)$로 변환된다.  
  - 이 표현은 이후 문서 임베딩과 비교되어 관련 문서를 탐색하는 데 사용된다.  

- **Retriever $p_\eta$ (Non-Parametric)**  
  - 벡터 공간 내에서 $q(x)$와 가장 유사한 문서 벡터 $d(z)$를 찾는다.  
  - 이때 **MIPS (Maximum Inner Product Search)** 알고리즘을 사용하여  
    내적이 가장 큰 문서들을 선택한다.  
  - 선택된 문서 예시:  
    - *“The middle ear includes the tympanic cavity and the three ossicles.”*  
    - *“This 14th century work is divided into 3 sections: ‘Inferno’, ‘Purgatorio’, ‘Paradiso’.”*  
  - 각각의 문서는 $z_1, z_2, \dots, z_k$ 로 표현된다.  

- **Generator $p_\theta$ (Parametric)**  
  - 선택된 문서들을 질의와 함께 입력받아,  
    문맥적으로 일관된 답변을 생성한다.  
  - 예시 생성 결과:  
    - Question Answering → “The middle ear includes the tympanic cavity and the three ossicles.”  
    - Fact Verification → “supports (y)”  
    - Jeopardy-style Question Generation → “This 14th century work is divided into 3 sections...”  

- **Marginalization 과정**  
  - 각 문서 $z_i$로부터 생성된 답변 확률 $p_\theta(y \mid x, z_i)$ 를  
    검색 확률 $p_\eta(z_i \mid x)$ 로 가중 평균하여 최종 응답을 산출한다.  
  - 이 과정을 통해 개별 문서의 정보 불확실성을 줄이고,  
    **가장 신뢰도 높은 종합적 답변**을 얻는다.  

#### 3. **End-to-End 학습 구조**  
- RAG는 **Retriever와 Generator를 동시에 학습**할 수 있는  
  **End-to-End Backpropagation** 구조를 가진다.  
- 학습 중에는 질의 인코더 $q(x)$와 생성기 $p_\theta$ 모두  
  손실 함수에 따라 함께 최적화된다.  
- 이를 통해 검색 정확도와 생성 품질을 동시에 향상시킬 수 있다.  

#### 4. **요약**  
- **Retriever**: 질의와 가장 관련 있는 문서를 확률적으로 검색 ($p_\eta(z \mid x)$)  
- **Generator**: 검색된 문서와 질의를 결합해 답변을 생성 ($p_\theta(y \mid x,z)$)  
- **Marginalization**: 여러 문서의 결과를 종합하여 최종 답변 도출  
- **장점**:  
  - 최신 정보 반영 가능  
  - 근거 문서에 기반한 **사실적(factual)** 응답 생성  
  - End-to-End 학습으로 **검색-생성 간 상호 최적화** 가능  

---

## p12. RAG

- RAG 시스템 아키텍처  

<img src="/assets/img/lecture/bigdatasearch/7/image_4.png" alt="image" width="800px">  

<a href="https://arxiv.org/pdf/2402.06196" target="_blank">Large Language Models: A Survey</a>

---

## p13. RAG 작동과정  

1. 사용자의 쿼리가 주어지면 **쿼리 인코더(Query Encoder)** 가 이를 **벡터 형태**로 변환한다.  
2. **지식 검색기(Retriever)** 가 인코딩된 쿼리를 바탕으로  
   **외부 지식 베이스(Vector Store)** 에서 관련 정보를 검색한다.  
3. 검색된 지식은 **지식 증강 생성기(Augmented Generator)** 의 입력으로 전달된다.  
4. 지식 증강 생성기는 검색된 지식을 활용하여  
   **사용자 쿼리에 대한 자연어 답변**을 생성한다.  

<img src="/assets/img/lecture/bigdatasearch/7/image_5.png" alt="image" width="800px">  

<a href="https://arxiv.org/abs/2312.10997?ref=pangyoalto.com" target="_blank">Retrieval-Augmented Generation for Large Language Models: A Survey(2023)</a>

---

## p14. RAG 아키텍처  

- **Retriever Types**

  - **일반 검색: Sparse Retrievers (SR)**  
    - 적용 용이  
    - 높은 효율성  
    - 우수한 성능  
    - 예: TF-IDF, BM25  

<img src="/assets/img/lecture/bigdatasearch/7/image_6.png" alt="image" width="600px"> 

  - **Dense Retrievers (DR)**  
    - 파인튜닝 허용  
    - 더 나은 적응력  
    - 다양한 검색 목적에 맞게 맞춤화 가능  
    - 예: DPR, Contriever  

<img src="/assets/img/lecture/bigdatasearch/7/image_7.png" alt="image" width="600px"> 

---

