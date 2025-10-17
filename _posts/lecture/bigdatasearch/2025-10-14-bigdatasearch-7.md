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

**생성형 AI 검색의 특징**

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

**주요 기술 요소**

- **LLM** : 생성형 AI 검색의 핵심 기술  
  - 질문의 의도 파악  
  - 검색 쿼리 최적화  
  - 정보 종합 및 답변 생성  
  - 자연스러운 문장 구성  

- **검색 증강 생성 (RAG: Retrieval Augmented Generation)**  
  - 외부 지식을 검색하여 LLM의 답변을 보강하는 기술  
  - 프로세스  
    - 사용자 질문 분석  
    - 관련 문서/정보 검색  
    - 검색된 정보를 컨텍스트로 제공  
    - LLM이 컨텍스트 기반 답변 생성  

- **의미기반 검색**  
  - 단순 키워드 매칭이 아닌 의미기반 검색 수행  
  - 기술  
    - 벡터 임베딩  
    - 유사도 계산: 질문과 문서 간 의미적 거리 측정  
    - 밀집 검색(Dense Retrieval)을 통한 정확도 향상  

---

## p5. 검색증강생성 (RAG: Retrieval Augmented Generation)  

**LLM의 근본적 한계**

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

## p8. RAG

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

**RAG 시스템 아키텍처**  

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

**RAG 시스템 아키텍처**  

<img src="/assets/img/lecture/bigdatasearch/7/image_4.png" alt="image" width="800px">  

<a href="https://arxiv.org/pdf/2402.06196" target="_blank">Large Language Models: A Survey</a>

---

## p13. RAG 작동과정  

1. 사용자의 쿼리가 주어지면 쿼리 인코더가 이를 벡터 형태로 변환한다.  
2. 지식 검색기가 인코딩된 쿼리를 바탕으로 외부 지식 베이스에서 관련 정보를 검색한다.  
3. 검색된 지식은 지식 증강 생성기의 입력으로 전달된다.  
4. 지식 증강 생성기는 검색된 지식을 활용하여 사용자 쿼리에 대한 자연어 답변을 생성한다.  

<img src="/assets/img/lecture/bigdatasearch/7/image_5.png" alt="image" width="800px">  

<a href="https://arxiv.org/abs/2312.10997?ref=pangyoalto.com" target="_blank">Retrieval-Augmented Generation for Large Language Models: A Survey(2023)</a>

---

## p14. RAG 아키텍처  

**Retriever Types**

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

## p15. RAG의 구조적 패러다임  

- 전통적 검색과 생성  

<img src="/assets/img/lecture/bigdatasearch/7/image_8.png" alt="image" width="240px"> 

- 검색 품질 향상  

<img src="/assets/img/lecture/bigdatasearch/7/image_9.png" alt="image" width="240px"> 

- 유사도 검색 모듈화, 패턴화  

<img src="/assets/img/lecture/bigdatasearch/7/image_10.png" alt="image" width="480px"> 

<a href="https://arxiv.org/abs/2312.10997?ref=pangyoalto.com" target="_blank">Retrieval-Augmented Generation for Large Language Models: A Survey(2023)</a>

---

## p16. RAG의 구조적 패러다임 변화  

- **Naïve RAG**  
  - 전통적 “Retrieve-Read” 구조 (인덱싱 → 검색 → 생성)  
  - **Augmentation**: 검색된 **단일 청크/문서**를 프롬프트에 직접 추가  
  - **장점**: 구현이 간단하고 비용 효율적  
  - **한계**: 검색 정확도·재현율(Recall) 부족, 환각·중복 발생, 맥락 통합 어려움  

- **Advanced RAG**  
  - 검색 대상, 검색 방법, 사전/사후 최적화 도입  
    - **인덱싱**: 슬라이딩 윈도우, 세분화, 메타데이터 활용  
    - **쿼리 최적화**: 쿼리 변환, 확장, 재작성  
    - **검색 후 처리**: 검색 결과 재순위화(rerank), 컨텍스트 압축  
  - **장점**  
    - 정보 과부하 방지, 핵심 정보 강조, 프롬프트 길이 제약 극복  
    - 정확도(Precision)와 재현율(Recall)을 향상시켜 LLM 출력 품질 개선  
  - **Augmentation**: **최적화된 청크**를 **여러 소스의 정보**로 정제하여 프롬프트에 추가  

- **Modular RAG**  
  - LLM 에이전트, 라우터 등 동적 모듈 사용  
  - **모듈화 및 유연성 강화**  
    - 검색, 생성, 필터링, 평가 등 다양한 기능 모듈 결합 가능  
    - 반복적/적응적 검색 및 End-to-End 통합 학습 지원  
    - 특정 도메인·작업에 맞춤화 용이  
  - **Augmentation**: **작업 흐름을 제어**하며 필요한 정보 검색·검증 및 복합적 증강 수행  

---

## p17. RAG의 구조적 패러다임  

**주요 Advanced RAG 유형**

| 유형 | 검색 단계의 개선 | 증강 프로세스에서의 차이점 |
|:--|:--|:--|
| **Pre-Retrieval<br>(Index)** | 데이터 전처리(청크 크기, 메타데이터 추가),<br>인덱스 구조 최적화 | 청크 자체의 품질을 높여 더 정확하고 관련성<br>높은 컨텍스트를 검색 |
| **Post-Retrieval<br>(Reranking)** | 검색된 상위 k개의 문서에 대해 Reranker 모델을<br>사용해 재순위 지정 | 가장 관련성 높은 문서만 선별하여<br>프롬프트에 포함<br>불필요한 노이즈(Noise) 감소 |
| **Query<br>Transformation** | 원래 쿼리를 LLM을 이용해 여러 쿼리로 확장<br>또는 HyDE(Hypothetical Document Embedding)<br>방식으로 변환 | 사용자의 의도를 더 잘 포착하는 다양한<br>컨텍스트를 검색하여 Recall(재현율) 향상 |
| **Multi-Hop RAG** | 한 번의 검색으로 답을 찾기 어려울 때, LLM이<br>여러 번의 검색·생성을 반복하여 정보를 결합 | 복합적인 추론이 필요한 경우, 단계별 중간 결과를<br>컨텍스트로 활용하여 최종 답변을 증강 |

---

## p18. RAG의 구조적 패러다임 변화  

**Modular RAG 유형: LLM 컨트롤러나 에이전트 활용, 동적인 판단 및 제어기능을 추가**

| 유형 | 제어/판단 방식 | 증강 프로세스에서의 차이점 |
|:--|:--|:--|
| **Fusion RAG** | 여러 검색엔진(벡터 검색 + 키워드 검색 등)<br>결과를 결합하여 사용 | 다양한 검색 소스의 정보를 통합, 컨텍스트 증강 |
| **Agentic RAG** | LLM이 에이전트 역할,<br>쿼리 분석 후 적절한 도구(검색, 코드 실행 등) 호출,<br>그 결과를 증강에 사용 | 단순 검색을 넘어 문제 해결 프로세스 전체를<br>증강된 컨텍스트로 활용, 복잡한 태스크에 적합 |
| **Corrective RAG** | LLM 생성 응답을 검증(Validation) 모듈로 평가,<br>필요하면 재검색을 수행하거나 응답을 수정 | 응답의 신뢰도와 정확성을 높이기 위해 피드백<br>루프를 통해 증강 과정을 반복 및 개선 |

Naive/Advanced RAG보다 **유연하고 지능적**인 증강을 제공

---

## p19. Agentic RAG  

[https://github.com/asinghcsu/AgenticRAG-Survey](https://github.com/asinghcsu/AgenticRAG-Survey){:target="_blank"}

<img src="/assets/img/lecture/bigdatasearch/7/image_11.png" alt="image" width="800px"> 

---

### 보충 설명  

#### 1. **Agentic RAG의 개요**  
- **Agentic RAG(에이전틱 RAG)** 는 기존 RAG의 한계를 극복하기 위해 **에이전트(Agent)** 개념을 도입한 확장형 구조이다.  
- 단순한 검색–생성(Retrieve–Generate) 절차를 넘어,  
  **도구 활용(Tool Use)**, **계획(Planning)**, **메모리(Memory)**, **역할(Task & Role)** 기반의 동적 의사결정을 수행한다.  
- 이를 통해 모델은 능동적으로 정보 탐색, 검증, 요약, 추론(reasoning)을 수행할 수 있다.  

#### 2. **기존 RAG의 한계 (RAG의 한계와 제약점)**  
- **환각(Hallucination)과 부정확성** : 비사실적 답변이 생성될 가능성 존재  
- **제한된 문맥 통합력** : 검색된 문서를 충분히 활용하지 못함  
- **정적인 워크플로우(Static Workflow)** : 검색–생성 과정이 고정되어 유연성 부족  
- **적응력 부족(Lack of Adaptability)** : 상황별 학습 및 전략 전환이 어려움  

#### 3. **Agentic RAG의 핵심 구조**  
- **에이전트(Agent)의 구성요소**  
  - **도구(Tools)** : 검색, 코드 실행, 데이터 분석 등 외부 기능 호출 가능  
  - **계획(Planning)** : 실행 순서를 결정하고, 필요한 정보 탐색 절차를 설계  
  - **역할(Task & Role)** : 각 에이전트의 목표와 담당 기능을 정의  
  - **메모리(Memory)** : 대화 이력 및 검색 결과를 장기적으로 저장·활용  

- **에이전틱 패턴(Agentic Pattern)**  
  - **자기반성(Reflection)** : 스스로 생성한 답변을 검토하고 수정  
  - **계획(Planning)** : 단계적 추론 및 반복적 개선 수행  
  - **다중 에이전트(Multi-Agents)** : 여러 에이전트가 협력적으로 문제를 해결  
  - **도구 활용(Tools)** : 각 작업에 적합한 툴을 동적으로 선택·호출  

- **에이전틱 워크플로우 패턴(Agentic Workflow Pattern)**  
  - **프롬프트 체이닝(Prompt Chaining)** : 이전 단계 결과를 다음 입력으로 연결  
  - **라우팅(Routing)** : 질의 유형에 따라 적절한 에이전트 또는 모듈로 분기  
  - **오케스트레이터–워커 구조(Orchestrator–Workers)** : 중앙 조정자가 각 에이전트의 작업을 관리  
  - **병렬화(Parallelization)** : 여러 작업을 동시에 실행하여 효율성 향상  
  - **평가–최적화(Evaluator–Optimizer)** : 생성된 응답을 평가하고 반복적으로 개선  

#### 4. **Agentic RAG의 장점 (이점과 향상된 기능)**  
- **의사결정 능력 향상** : 에이전트가 상황을 분석하고 스스로 판단 가능  
- **다단계 추론 및 반복적 개선(Multi-step reasoning)** : 복잡한 문제를 단계별로 해결  
- **실시간 데이터 통합(Real-time data integration)** : 최신 정보 반영  
- **적응형 학습 및 문맥 인식(Adaptive learning & Context awareness)** : 문맥 변화에 따라 전략을 조정  
- **정확도 향상 및 환각 감소** : 사실 기반 응답을 생성하고 오류를 줄임  

#### 5. **Agentic RAG의 분류 (Taxonomy)**  
- **단일 라우터(Single Router)** : 하나의 중앙 조정자가 전체 과정을 통제  
- **다중 에이전트(Multi-Agent)** : 여러 에이전트가 병렬로 협력  
- **계층형 구조(Hierarchical)** : 상위 조정자가 하위 에이전트를 관리  
- **교정형(Corrective)** : 응답을 검증하고 필요 시 수정  
- **적응형(Adaptive)** : 상황에 따라 검색·생성 전략을 자동 조정  
- **Agent-G / GeAR** : 강화학습 기반 고도화된 에이전트형 RAG  
- **문서형 워크플로우(Agentic Document Workflow)** : 문서 기반 분석과 검증 자동화  

#### 6. **프레임워크 및 기술 스택 (Framework and Stack)**  
- 대표적인 Agentic RAG 개발 프레임워크  
  - **LangGraph**, **LlamaIndex**, **AG2 (AutoGen)**, **Semantic Kernel**, **Crew AI**, **Amazon Bedrock**, **OpenAI Swarm**  
- 이들은 **모듈형 설계(Modular Design)**, **LLM 에이전트 오케스트레이션(Orchestration)**,  
  **실시간 데이터 연동(Real-time Integration)**, **다중 에이전트 협력(Multi-agent Collaboration)** 을 지원하여  
  Agentic RAG의 **지능형 자동화(Intelligent Orchestration)** 를 가능하게 한다.  

#### 7. **요약**  
- Agentic RAG는 기존 RAG의 한계를 극복하고  
  **자기반성(Self-reflection)**, **도구 활용(Tool Use)**, **계획(Planning)** 기능을 결합한  
  **지능형 검색–생성 패러다임(Intelligent Retrieve–Generate Paradigm)** 으로 발전하였다.  
- 단순한 검색 보조를 넘어,  
  **자율적 의사결정이 가능한 에이전트 기반 검색 시스템(Agent-based Search System)** 으로 진화하고 있다.  

---

## p20. RAG의 핵심 구성요소  

- **문서처리와 인덱싱(Indexing) 파이프라인(오프라인)**  
  - 데이터 수집  
  - 문서 파싱 및 청킹  
    - **청킹전략**  
      - **고정 크기 청킹**: 일정한 토큰/문자 수로 분할  
      - **의미 단위 청킹**: 문단, 섹션 단위로 분할  
      - **슬라이딩 윈도우**: 오버랩을 두고 분할  
      - **문서 구조 기반**: 제목, 목차 구조 활용  
  - 임베딩 벡터 생성  
  - 벡터 DB 저장  

- **검색 파이프라인(온라인)**  
  - 쿼리 임베딩  
  - 유사도 검색  
  - 리랭킹(선택적)  
  - 컨텍스트 구성  

- **생성 파이프라인**  
  - 프롬프트 템플릿  

**프롬프트 템플릿**  

```

다음 문서들을 참고하여 질문에 답변하세요.  

=== 참고 문서 ===
{context}
===============

질문: {query}

답변 시 다음을 준수하세요:
제공된 문서의 정보만 사용하세요
정보가 없으면 "문서에서 찾을 수 없습니다"라고 답하세요
각 정보의 출처 번호를 명시하세요

답변:
```

---

## p21. RAG 구현 과정  

**RAG (Retrieval-Augmented Generation) 파이프라인**  
- 기존의 언어 모델에 검색 기능을 추가  
- 주어진 질문이나 문제에 대해 더 정확하고 풍부한 정보를 기반으로 답변 생성 가능  

- **파이프라인은 크게 5단계로 구성**

  ① **데이터 로드**: RAG에 사용할 **외부 데이터**를 필요한 형식으로 변환하여 불러오는 단계  

  ② **텍스트 분할**: 불러온 데이터를 작은 크기의 단위(**chunk**)로 분할하는 과정  

  ③ **인덱싱**: 분할된 텍스트를 **벡터 형태로 임베딩**하고 색인하는 단계 (검색 정확도와 효율을 고려)  

  ④ **검색**: 사용자의 질문이나 주어진 **context**에 가장 적합한 정보를 찾아내는 과정  

  ⑤ **생성**: 검색된 정보를 바탕으로 사용자의 **질문에 최적 답변을 생성**하는 최종 단계  

---

## p22~23. RAG 구현 과정  

<img src="/assets/img/lecture/bigdatasearch/7/image_12.png" alt="image" width="800px">  
<img src="/assets/img/lecture/bigdatasearch/7/image_13.png" alt="image" width="800px">  

---

## p24. RAG 개발 시 필요한 것  

- **데이터**  
  - 다양한 형식의 문서(CSV, JSON, PDF, XML 등)와 DocumentLoader  
  - DB 및 웹 문서 크롤러  
  - 문서를 작은 단위로 분할할 Text Splitter  

- **임베딩 모델**  
  - 텍스트를 덴스 벡터로 변환할 모델  
  - Sentence-transformers, OpenAI 임베딩 모델, HuggingFaceEmbeddings 등  

- **벡터 데이터 베이스**  
  - 벡터를 저장하는 저장소  
  - 벡터 유사도 검색 및 메타 데이터 검색 기능 제공  
  - Pinecone, Qdrant, Chroma, ElasticSearch, FAISS 등  

- **LLM 애플리케이션 개발 프레임워크**  
  - LangChain / LangGraph  

---

## p25. Document Loader  

- **다양한 소스에서 문서를 불러오고 처리하는 과정을 담당**  
  - **다양한 소스 지원**  
    - PDF / 웹페이지 / 데이터베이스 등 다양한 소스에서 문서를 불러올 수 있음.  
  - **데이터 변환 및 정제**  
    - LangChain의 module이나 algorithm이 처리하기 쉬운 형태로 변환.  
  - **효율적인 데이터 관리**  
    - 대량의 문서 데이터를 효율적으로 관리 및 접근 가능하게 함.  

- **종류**  
  - WebBaseLoader : 웹페이지 문서를 로드하여 필요한 텍스트를 추출 변환함  
  - TextLoader : 텍스트 파일을 로드함  
  - DirectoryLoader : 디렉토리 내의 모든 문서를 로드함.  
  - CSVLoader : csv파일에서 데이터를 loader  

---

## p26. WebBaseLoader

<img src="/assets/img/lecture/bigdatasearch/7/image_14.png" alt="image" width="800px">  

---

### 보충 설명  

#### 1. **WebBaseLoader의 역할**  
- `WebBaseLoader`는 웹페이지의 HTML 문서에서 텍스트 데이터를 불러와,  
  **LLM이 학습하거나 검색할 수 있는 형태로 변환**하는 모듈이다.  
- LangChain의 `document_loaders`에 포함된 도구로,  
  웹페이지의 **특정 영역(HTML 클래스나 태그 단위)** 만 선택적으로 파싱할 수 있다.  

#### 2. **SoupStrainer를 활용한 선택적 파싱**  
- `bs4.SoupStrainer`는 BeautifulSoup의 기능으로,  
  전체 HTML이 아닌 **특정 클래스나 태그에 해당하는 부분만 읽어들여**  
  처리 속도를 높이고 불필요한 데이터 로딩을 방지한다.  
- 코드에서 `class_=("article-header", "article-content")`는  
  기사 제목(`header`)과 본문(`content`)만 추출하도록 지정한 것이다.  

#### 3. **코드 동작 요약**  
1. `url1`, `url2`에 지정된 두 개의 블로그 페이지를 대상으로 크롤링 수행  
2. `WebBaseLoader`가 지정된 클래스의 HTML 요소만 파싱  
3. `loader.load()`가 문서를 불러와 `docs` 객체에 저장  
4. `len(docs)`를 통해 로드된 문서의 개수를 확인  

#### 4. **활용 예시**  
- 뉴스, 블로그, 위키 문서 등에서 필요한 텍스트만 자동 추출하여  
  **RAG 파이프라인의 인덱싱 단계에 직접 활용**할 수 있다.  
- 특히 LLM 응용에서는 웹 최신 정보를 반영하거나  
  외부 지식을 실시간으로 가져올 때 유용하다.  

---

## p27. PDF Loader  

- [PyPDFLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.pdf.PyPDFLoader.html){:target="_blank"}  
  - PDF 문서 페이지별로 로드, 텍스트를 추출하여 documents list 객체로 반환  

- [PyMuPDFLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.pdf.PyMuPDFLoader.html#langchain_community.document_loaders.pdf.PyMuPDFLoader){:target="_blank"}  
  - PDF 파일의 페이지를 로드하고, 각 페이지를 개별 document 객체로 추출  
  - 자세한 메타데이터 추출도 가능  

- [PyPDFDirectoryLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.pdf.PyPDFDirectoryLoader.html#langchain_community.document_loaders.pdf.PyPDFDirectoryLoader){:target="_blank"}  
  - 특정 폴더에 있는 모든 PDF 파일을 가져옴  

- [OnlinePDFLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.pdf.OnlinePDFLoader.html#langchain-community-document-loaders-pdf-onlinepdfloader){:target="_blank"}  
  - 온라인 PDF 파일 데이터를 가져옴  

---

## p28. PyPDFLoader vs. PyMuPDFLoader

<img src="/assets/img/lecture/bigdatasearch/7/image_15.png" alt="image" width="800px">  

**주로 PDF의 메타데이터를 추출하고 관리하는 영역에서 차이가 있음**

---

### 보충 설명  

#### 1. **PyPDFLoader와 PyMuPDFLoader의 공통점**  
- 두 로더 모두 PDF 문서를 **페이지 단위로 로드**하고 각 페이지를 LangChain의 `Document` 객체로 변환함.  
- `page_content`로 **본문 텍스트**를, `metadata`로 **부가 정보**를 확인할 수 있음.  

#### 2. **PyPDFLoader의 특징**  
- 표준 PyPDF 기반.  
- 각 페이지의 **텍스트**와 **페이지 번호/경로** 등 **단순 메타데이터** 위주 제공 → **텍스트 분석 중심 작업**에 적합.  
- 예시 출력: `{'source': '/content/Samsung_Electronics_Sustainability_Report_2023_KOR.pdf', 'page': 0}`

#### 3. **PyMuPDFLoader의 특징**  
- PyMuPDF(fitz) 기반.  
- 파일 경로, **전체 페이지 수**, **PDF 포맷 버전**, **생성/수정 일자**, **제작 도구(예: Adobe InDesign)** 등 **풍부한 메타데이터**를 제공 → **문서 관리/출처 추적/버전 관리**에 유용.  
- 예시 출력: `{'source': '/content/Samsung_Electronics_Sustainability_Report_2023_KOR.pdf', 'file_path': '/content/Samsung_Electronics_Sustainability_Report_2023_KOR.pdf', 'page': 0, 'total_pages': 129, 'format': 'PDF 1.7', 'creator': 'Adobe InDesign 15.1 (Macintosh)', 'creationDate': 'D:20230707144723+09\'00\'', ...}`

#### 4. **요약 비교**

| 구분 | PyPDFLoader | PyMuPDFLoader |
|:--|:--|:--|
| 기반 라이브러리 | PyPDF | PyMuPDF(fitz) |
| 텍스트 추출 | 가능 | 가능 |
| 메타데이터 | 단순(경로·페이지) | 상세(제작정보·생성일·포맷 등) |
| 적합 용도 | 텍스트 기반 분석 | 문서 관리·구조 정보 추출 |

---

## p29. Text Splitter  

문서들을 **효율적으로 처리**하고, 시스템이 정보를 보다 잘 활용할 수 있도록  
크고 복잡한 문서를 LLM이 받아들일 수 있는 **효율적인 작은 규모의 단위(Chunk)** 로 나눔  

- **분할의 필요성**  
  - **관련성 높은 정보 검색으로 정확성 향상**  
    - 문서를 세분화함으로써 **Query와 연관성이 있는 정보만** 가져와 특정 주제나 내용에 초점을 맞춤  
  - **리소스 최적화(효율성)**  
    - **LLM의 입력 토큰 개수가 제한**되어 있으므로 전체 문서를 입력하면 **비용이 많이 발생**  
    - 많은 정보 속에서 발췌하다 보면 **Hallucination 문제를 유발하므로 답변에 필요한 정보만 효율적으로 발췌**  

- **고려사항**  
  - 텍스트 분리 시 독립적으로 의미를 지닌 단위로 분리: 문장, 구절, 단락 등  
  - 청크 크기  
    - LLM 모델의 입력 크기 및 비용 등을 고려하여 최적 크기를 결정  
    - 단어 수, 문자 수 등을 기준으로 크기를 측정  

---

## p30. TextSplitter  

- **CharacterTextSplitter**  

- **RecursiveCharacterTextSplitter**  

- **tiktoken (token-based chunking)**  

---

## p31. TextSplitter  

- CharacterTextSplitter
  - 문서를 분할구분자 기준으로 나눔
  - 문자 중복을 허용하여 문맥을 유지하고 정보손실을 최소화

<img src="/assets/img/lecture/bigdatasearch/7/image_16.png" alt="image" width="600px"> 

- RecursiveCharacterTextSplitter
  - 텍스트를 재귀적으로 분할, 의미적으로 관련있는 텍스트 조각들로 나눔
  - 문자 리스트 [‘\n\n’, ‘\n’, ‘ ’, ‘’] 의 문자를 순서대로 사용하여 텍스트 분할

<img src="/assets/img/lecture/bigdatasearch/7/image_17.png" alt="image" width="540px"> 

---

## p32. TextSplitter  

- CharacterTextSplitter

<img src="/assets/img/lecture/bigdatasearch/7/image_18.png" alt="image" width="600px"> 

- RecursiveCharacterTextSplitter

<img src="/assets/img/lecture/bigdatasearch/7/image_19.png" alt="image" width="600px"> 

---

## p33. tiktoken(token-based chunking)

- **텍스트를 토큰 수 기준으로 청크를 나눔**  
  - LLM에 적용된 Tokenizer를 사용하면 모델 입력 토큰 수 조절이 가능  
  - OpenAI API의 경우, **BPE 토크나이저로 만들어진 tiktoken** 라이브러리를 지원  
  - open-source model의 경우 model별 tokenizer 적용  

- **OpenAI의 tiktoken API 사용**  
  - **Parameters**  
    - chunk_size: 각 청크의 최대 길이  
    - chunk_overlap: 분할된 텍스트 조각들 사이에서 중복으로 포함될 문자 수  
    - encoding_name: 텍스트를 토큰으로 변환하는 인코딩 방식 (cl100k_base: ada-002 model 사용)

<img src="/assets/img/lecture/bigdatasearch/7/image_20.png" alt="image" width="800px"> 

---

## p34. 임베딩 모델  

- **임베딩?**  
  - 텍스트를 고차원 벡터 공간의 점으로 변환하는 것  

```
"강아지": [0.8, 0.6, 0.1, ..., 0.3]
"개": [0.79, 0.61, 0.09, ..., 0.31] (유사)
"자동차": [0.1, -0.3, 0.9, ..., -0.2] (다름)
```

- **임베딩 모델**  
  - 트랜스포머(BERT, RoBERTa 등) 아키텍처 기반, 특히 검색 작업에 특화된 **이중 인코더 구조**를 사용  
  - 이중 인코더 구조를 사용  
    - 쿼리(Q)를 벡터 $v_Q$로 변환하는 Query Encoder와 문서 청크(D)를 벡터 $v_D$로 변환하는 문서 인코더로  
      Q와 D를 개별적으로 인코딩하고, 벡터 DB에 미리 저장하므로 검색 속도가 빠름

  - 임베딩 모델의 학습: Contrastive Learning  
    - “**유사한 것은 가깝게, 유사하지 않은 것은 멀리**” 벡터 공간에 배치하도록 모델을 학습  
    - 모델이 쿼리(Q)와 **정답 문서(D⁺)** 쌍에 대해서는 높은 유사도를,  
      쿼리와 **오답 문서(D⁻)** 쌍에 대해서는 낮은 유사도를 출력하도록 학습  

---

## p35. 임베딩 모델  

RAG 시스템은 속도를 위해 이중 인코더를 사용하여 후보군을 빠르게 검색한 후,  
정확도를 위해 크로스 인코더를 사용하여 순위를 재조정하는 **하이브리드 파이프라인**을 주로 사용  

| **특징** | **이중 인코더 (Dual-Encoder)** | **크로스 인코더 (Cross-Encoder)** |
|-----------|-----------------------------|----------------------------------|
| **구조** | 쿼리와 문서를 각각 인코딩<br>$(Q \rightarrow v_Q,\ D \rightarrow v_D)$ | 쿼리와 문서를 연결하여 함께 인코딩<br>$([Q;D] \rightarrow \text{Score})$ |
| **학습/추론** | 분리 인코딩 → 벡터 저장 → 코사인 유사도 | 결합 인코딩 → 단일 점수 출력 (관련도) |
| **속도** | 매우 빠름 (벡터 검색) | 매우 느림 (쿼리당 문서별 인코딩 필요) |
| **정확도** | 높음 (대조 학습으로 충분히 학습) | 최고 (문맥적 상호작용 분석 가능) |
| **RAG 활용** | 1단계 검색 (Retrieval):<br>수십억 개 문서 중 *k*개 후보군 선별 | 2단계 재순위 지정 (Reranking):<br>*k*개 후보군 순위 재조정 |

---

## p36. Embedding  

- **모델 제공자**  
  - Google, OpenAI, Cohere, Amazon: 상용 임베딩 모델  
  - Hugging Face: Transformers 라이브러리를 통해 오픈소스 임베딩 모델을 제공  

- **임베딩 메소드**  
  - **embed_documents**  
    - 입력: 문서 객체의 집합  
    - 각 문서를 벡터 공간에 임베딩, 대량의 텍스트 데이터를 배치 단위로 처리할 때 사용  
  - **embed_query**  
    - 입력: 단일 텍스트 쿼리  
    - 쿼리를 벡터 공간에 임베딩  

- **유료 임베딩 모델** (OpenAI, Google, Cohere, Amazon)  
  - 사용하기 편리하지만 비용 발생  
  - API 통신을 이용하므로 보안 우려 존재  
  - 한국어 포함 다국어 임베딩 지원  
  - GPU 없이도 빠른 임베딩 가능  

- **로컬 임베딩 모델** (HuggingFace)  
  - 무료이지만 다소 어려운 사용  
  - 오픈소스 모델 사용으로 보안 우수  
  - 모델마다 지원 언어가 다름  
  - GPU 없을 시 임베딩 속도 느림  

> 임베딩 모델에 따라 **임베딩 벡터의 값과 차원**이 다르게 표현되므로,  
> **용도에 맞는 임베딩 모델** 선택이 중요!

---

## p37. OpenAIEmbeddings  

- **OpenAI의 API를 활용하여**, 각 문서를 대응하는 **임베딩 벡터로 변환**  

<img src="/assets/img/lecture/bigdatasearch/7/image_21.png" alt="image" width="800px"> 

---

## p38. HuggingFaceEmbeddings  

- **Hugging Face의 트랜스포머 모델**을 사용하여 문서/문장을 임베딩하는 데 사용  
  - **Hugging Face 모델 허브**에서 사전 훈련된 임베딩 모델을 다운로드 받아서 적용 가능  

<img src="/assets/img/lecture/bigdatasearch/7/image_22.png" alt="image" width="800px"> 

---

## p39. 임베딩 모델  

- **임베딩 모델 선택**  
  - **오픈소스:** 비용 절감, 데이터 프라이버시, 모델 커스터마이징 측면에서 큰 이점  

| 구분 | 모델 예시 | 임베딩 차원 | 특징 및 장점 | 주요 사용 사례 |
|------|------------|--------------|----------------|----------------|
| 상용 (API) | OpenAI<br>text-embedding-3-large | 3072<br>(API로 축소 가능) | **최상위 성능**, 다국어 능력 우수,<br>차원축소 기능으로 비용/속도 최적화 가능,<br>관리 용이 | 고정밀이 요구되는 전문 지식 기반 RAG,<br>초기 프로토타이핑, API 종량제 모델에 적합 |
| | Cohere Embed v3/v4 | 주로 1024 | **긴 컨텍스트(최대 128K)** 처리 강점,<br>**다국어 및 멀티모달 지원** | 긴 법률 문서, 연구 논문 등 **장문 검색이 필수인 분야**,<br>**긴 텍스트 청크**를 기반으로 하는 RAG |
| 오픈소스 | BAAI BGE-M3 | 1024 | **다기능성 (Dense + Sparse)** 지원,<br>다국어 성능 우수, **하이브리드 검색**에 최적화,<br>커스터마이징 용이 | 고성능/하이브리드 검색이 필요한 복합 RAG,<br> 사내 보안 및 통제 환경 |
| | e5 계열<br>(e5-large, multilingual-e5) | 768 / 1024 | **균형 잡힌 성능과 효율성**, **SOTA에 근접한 성능**을<br>제공하는 대표적 오픈소스 모델 | 준수한 성능 대비 **리소스 효율**이 중요한<br>대규모 인덱싱, 일반적인 RAG |
| | MiniLM 계열<br>(all-MiniLM-L6-v2) | 384 | **초경량, 고속 추론**, 낮은 차원으로<br>벡터 DB 저장 공간 및 **검색 속도 최적화**,<br>CPU 환경 적합 | 낮은 지연 시간(Low Latency)이 요구되는<br>애플리케이션, 리소스 제한적인 환경 |

---

## p40. 임베딩 모델  

**RAG 개발 시 임베딩 모델 선택 가이드**  

- **정확도 최우선**  
  - **OpenAI / Cohere** 같은 상용 API 사용하거나 **BGE-M3**와 같은 최신 고성능 오픈소스 모델을 선택  
  - Reranker(재순위 모델)을 추가하여 성능 향상  

- **비용/효율성 최우선**  
  - **all-MiniLM-L6-v2** 또는 e5-base와 같은 경량 모델을 선택  
  - 낮은 차원(384d, 768d)을 사용하면 인프라 비용을 크게 절감  

- **데이터 프라이버시 / 커스터마이징**  
  - **BGE-M3 또는 E5** 같은 오픈소스 모델을 자체 서버에 배포  
  - **자체 도메인 데이터로 파인튜닝**하여 보안 및 성능을 확보  

- **긴 문서 처리**  
  - **Cohere**나 컨텍스트 길이가 긴 오픈소스 모델(예: **Qwen3-Embedding**)을 고려  

---

## p41. 유사도 계산  

- **코사인 유사도(Cosine Similarity)**  
  similarity = cosine_similarity(vector1, vector2)  
  범위: -1 ~ 1 (1에 가까울수록 유사)  

- **유클리드 거리(Euclidean Distance)**  
  distance = euclidean_distance(vector1, vector2)  
  범위: 0 ~ ∞ (0에 가까울수록 유사)  

- **내적 (Dot Product)**  
  score = dot_product(vector1, vector2)  

| 방식 | 수식 / 정의 | 특징 | 활용 예시 |
|------|---------------|-------|------------|
| **Cosine Similarity** | cos(θ) = (A·B) / (‖A‖‖B‖) | 방향(의미적 유사성)에 민감 | 텍스트 임베딩 기본 |
| **Euclidean Distance** | d = √Σ(xᵢ - yᵢ)² | 절대 거리 기준 | 이미지 벡터, 위치 기반 |
| **Dot Product** | A·B = Σ(aᵢbᵢ) | 크기 + 방향 반영 | 신경망 내부 표현 |
| **Hamming Distance** | bit 간 불일치 수 | 이진 벡터 전용 | 해시 기반 검색 |

---

## p42. 벡터 데이터베이스  

- **벡터 데이터베이스(Vector Database, Vector DB)**  
  - RAG(검색 증강 생성) 시스템에서 외부 지식을 저장하고 검색하는 핵심 인프라  
  - 고차원 벡터의 저장, 인덱싱 및 유사도 검색에 최적화  

| 특징 | 설명 |
|------|------|
| **벡터 스토리지** | 텍스트, 이미지, 오디오 등 복잡한 데이터를 수치화한 임베딩 벡터와<br>해당 벡터의 메타데이터를 함께 저장 |
| **유사도 검색 (ANN)** | 모든 벡터와의 비교 검색(O(N))이 아닌<br>**Approximate Nearest Neighbor(근사 최근접 이웃)** 알고리즘(예: HNSW, IVF)을 사용하여<br>수십억 개의 벡터 중 가장 유사한 벡터를 **초고속**으로 검색 |
| **확장성** | 대규모 데이터셋과 쿼리 부하에 대응하기 위해<br>**수평 확장(Horizontal Scaling)** 을 기본적으로 지원 |
| **필터링** | 벡터 검색과 함께 메타데이터(예: 작성자, 날짜, 카테고리) 기반으로 **필터링**이 가능 |

---

## p42. 벡터 데이터베이스  

- **벡터 데이터베이스(Vector Database, Vector DB)**  
  - RAG(검색 증강 생성) 시스템에서 외부 지식을 저장하고 검색하는 핵심 인프라  
  - **고차원 벡터**의 저장, 인덱싱 및 유사도 검색에 최적화  

| 특징 | 설명 |
|------|------|
| **벡터 스토리지** | 텍스트, 이미지, 오디오 등 복잡한 데이터를 수치화한 **임베딩 벡터**와<br>해당 벡터의 **메타데이터**를 함께 저장 |
| **유사도 검색 (ANN)** | **모든 벡터와의 비교 검색(O(N))이 아닌**<br>**Approximate Nearest Neighbor(근사 최근접 이웃)** 알고리즘(예: HNSW, IVF)을 사용하여<br>수십억 개의 벡터 중 가장 유사한 벡터를 **초고속**으로 검색 |
| **확장성** | 대규모 데이터셋과 쿼리 부하에 대응하기 위해 수평 확장(Horizontal Scaling)을<br>기본적으로 지원 |
| **필터링** | 벡터 검색과 함께 메타데이터(예: 작성자, 날짜, 카테고리) 기반으로 **필터링**이 가능 |

---

## p43. 벡터 데이터베이스  

**상용 vs. 오픈소스 벡터 DB**  

- **상용 벡터DB**  
  - Pinecone, Vertex AI Vector Search (Google), Azure AI Search (Microsoft),  
    Amazon OpenSearch Service (w/ k-NN) (AWS)  
  - **특징:** 완전 관리형(Fully Managed), 클라우드 기반 서비스  

  - **장점**  
    - 인프라 관리 부담 해소: 설치 확장, 백업, 모니터링 등 운영 오버헤드가 없음  
    - 고가용성 및 성능 보장: 안정적인 서비스 가능  
    - 전문지원: SLA(서비스 수준 계약) 기반의 기술지원  
    - Time-to-Market  

  - **단점**  
    - 비용: 벡터수, 차원, 쿼리량에 따라 비용 증가  
    - 벤더 종속성  
    - 커스터마이징 제한: 인덱싱 알고리즘, 하부 인프라 등의 세부적인 제어 불가능  
    - 데이터 주권/보안 우려  

  - **주요 활용 예:** 실시간성이 중요한 프로덕션 환경, 빠른 Prototyping, 인프라 운영 인력 부족 시  

---

## p44. 벡터 데이터베이스  

- **주요 상용 벡터 DB 특징 요약**

| 구분 | 주요 서비스 | 핵심 특징 | 하이브리드 검색 | 적합한 사용자 |
|------|--------------|-------------|------------------|----------------|
| **전용 벡터 DB** | Pinecone (대표적) | 벡터 검색에 최적화된 아키텍처<br> 다양한 LLM/임베딩 API와 연동용이<br> 클라우드 agnostic (AWS, GCP 등 선택 가능) | 메타데이터 필터링 및 희소 벡터(Sparse) 지원 | 고성능 및 낮은 지연 시간이 필수적인 RAG 개발자, <br> 자체 DB 운영을 원치 않는 기업 |
| **클라우드 통합** | Vertex AI Vector Search<br>(Google) | Google Cloud 서비스와 긴밀한 통합 <br> 강력한 ML 플랫폼(Vertex AI)의 일부 | 메타데이터 필터링 지원 | Google Cloud를 주력으로 사용. <br> ML Ops 환경을 통합하려는 기업 |
|  | Azure AI Search<br>(Microsoft) | 기존 Full-Text Search 기능과의 강력한 결합. <br> Microsoft 생태계(Azure, OpenAI) 통합 | **네이티브 하이브리드 검색** <br>(Full-Text + Vector) 강력 지원 | Microsoft Azure를 사용하거나, <br> 하이브리드 검색을 중요하게 생각하는 기업 |
|  | Amazon OpenSearch<br>Service (w/ k-NN)<br>(AWS) | 로그 및 검색 기능 중심. <br> 벡터 검색을 기존 검색 엔진 기능에 추가 통합 | - | AWS를 사용하며, <br> 기존 OpenSearch 클러스터에 벡터 기능을 추가하려는 기업 |

---

## p45. 벡터 데이터베이스  

- **오픈소스 벡터 DB**  
  - Milvus, Qdrant, Weaviate, Chroma, pgvector (PostgreSQL 확장), FAISS, ElasticSearch  

  - **특징**  
    - **자유로운 라이선스**, 자체 구축 및 운영 필요 (Self-hosted)  

  - **장점**  
    - 비용 절감  
    - 최대 유연성 및 통제: 소스코드 수정 및 커스터마이징 가능  
    - 데이터 프라이버시: 민감한 기업 내부 데이터를 외부에 전송하지 않고 관리  

  - **단점**  
    - 운영 복잡성  
    - 기술지원  

  - **활용사례**  
    - **대규모 데이터 인프라를 이미 보유**한 기업  
    - **높은 수준의 데이터 보안**이 요구되는 금융/공공 기관  
    - **도메인 특화된 기능**을 개발해야 하는 경우  

---

## p46. 벡터 데이터베이스  

**Indexing 알고리즘**  

- **Flat Index (No optimization)**  
  - Indexing 기법 없이 벡터를 저장하는 방법  
  - 저장된 모든 벡터들과 유사도를 계산, 가장 높은 유사도를 지닌 벡터를 찾는 방법  
  - 10000~50000개 정도의 벡터에서 적당한 성능과 높은 확장성  

- **PQ (Product Quantization)**  
  - 원벡터를 균등하게 몇 개의 서브 벡터로 쪼개고,  
    각 서브 벡터들을 Quantization하여 크기를 줄임  
  - 빠르고 정확도가 높고, 큰 데이터 셋에서 사용하기 좋은 기법  

- **LSH (Locality-Sensitive Hashing)**  
  - 벡터들을 Hashing 함수를 이용해 Bucket에 매핑하는 방법  
  - 유사도 검색과 색인을 모두 동일한 Hashing 함수를 사용하여  
    같은 Bucket에 있는 벡터들과 비교 및 탐색  
  - Bucket에 있는 데이터들 사이에서만 유사도 검색을 하므로 검색속도가 빠름  

---

## p47. 벡터 데이터베이스  

**HNSW (Hierarchical Navigable Small World graph)**  

- **탐색 가능한 작은 세계(NSW)** 그래프 구조를 **계층적으로** 구성하여,  
  방대한 데이터셋에서 쿼리 벡터와 가장 유사한 벡터를 **빠르게** 찾아내는 것을 목표로 함  

  - **작은 세계 (Small World)**: 임의의 두 노드 사이의 거리가 매우 짧은 그래프를 의미  

  - **계층 구조 (Hierarchy)**: 그래프를 여러 레벨(층)으로 나누어 검색 속도와 정확도를 조절  
    - 상위 레벨: 매우 희소한 그래프 → 전역 탐색 (빠르게 후보 집합을 찾음)  
    - 하위 레벨: 상세(밀집) 그래프 → 지역적으로 정확한 최근접 탐색  

- **HNSW 그래프 구조**: HNSW 인덱스는 다중 레벨 그래프(Multilayer Graph)와 진입점(Entry Point)으로 구성  

  - **다중 레벨 그래프 (Multilayer Graph)**  
    새로운 벡터(v)가 인덱스에 삽입될 때, 무작위로 레벨 L이 할당  

    - **레벨 0 (Base Layer)**  
      - 모든 벡터 노드를 포함하며, 가장 촘촘하게 연결되어 **정확한 지역 검색** 수행  
        모든 탐색은 최종적으로 이 레벨에서 마무리  

    - **상위 레벨 (L > 0)**  
      - 벡터의 부분 집합만을 포함한 매우 희소한 그래프  
        → 노드 수가 적고 연결이 성기기 때문에 **빠른 전역 탐색**을 위해 사용  

    - **연결 제한 (M)**  
      - 각 노드가 갖는 최대 이웃의 개수로, M 값이 클수록 정확도는 높아지지만, 메모리 사용량과 인덱스 구축 시간 증가  

  - **진입점 (Entry Point, E)**  
    - 인덱스에서 **가장 높은 레벨**에 있는 임의의 노드를 **진입점**으로 설정. 모든 검색은 이 E 노드에서 시작  

---

## p48. 벡터 데이터베이스  

**HNSW (Hierarchical Navigable Small World graph)**  

- **탐색 알고리즘**  

  - **시작**:  
    entry_point와 현재 그래프의 최상위 레벨에서 시작  

  - **상위 레벨에서 greedy search**  
    - 현재 노드에서 인접 노드들 중 쿼리와 거리(또는 유사도)가 더 나은 노드가 있으면 그 쪽으로 이동  
    - 더 이상 개선 불가할 때까지 반복 → 로컬 최적점 도달  

  - **하위 레벨로 내려가기**  
    - 해당 레벨에서 탐색 후보 집합(efSearch)을 이용한 **beam-like search**  
      (best-first 후보 유지)를 실행  
    - 하위 레벨 시작점은 상위 레벨에서의 최종 노드(로컬 최적점)  

  - **최저 레벨 (0)** 에서 최종 Top-K 반환. 선택적으로 **re-ranking** 수행  

```
function search(query, efSearch, entry_point, max_level):
    cur_node = entry_point
    # greedy search on upper levels (max_level .. 1)
    for level in range(max_level, 0, -1):
        improved = True
        while improved:
            improved = False
            for neighbor in cur_node.neighbors[level]:
                if dist(query, neighbor.vector) < dist(query, cur_node.vector):
                    cur_node = neighbor
                    improved = True

    # at level 0: do best-first search with efSearch
    candidates = priority_queue()
    visited = set()
    add cur_node to candidates
    top_candidates = best_first_search(query, candidates, efSearch)
    return top_candidates.topK()
```

---

## p49. 벡터 데이터베이스  

**IVF (Inverted File index)**  
- 클러스터링을 통해 검색 범위를 축소함으로써 검색 효율을 높이는 방법:  
  **Clustering + Inverted File**  
  - 대규모 벡터 집합의 ANN(근사 최근접 탐색)을 위해  
    **데이터를 여러 버킷(list)으로 군집화**  
    (e.g. k-means 등을 적용해 N개 클러스터 구성)  
  - 검색 쿼리가 주어지면, 쿼리가 포함된 cluster를 찾고,  
    해당 cluster의 inverted list 내 벡터들에 대해 유사도 검색 수행  

- **IVFADC**  
  - IVF에 **Product Quantization (PQ)** 을 결합해  
    검색 메모리와 계산 비용을 줄인 기법의 표준  

- **알고리즘**  
  - 전체 벡터들을 nlist개의 군집(버킷)으로 나눈다 (코어 센트로이드 $C_i$)  
  - 각 벡터 $x$는 그 벡터가 속한 centroid의 버킷에 저장  
    (또는 residual을 PQ로 압축하여 저장)  
  - 검색 시 쿼리 $q$에 대해 가장 가까운 nprobe개의 centroid를 선택 —  
    해당 버킷들만 탐색  
    - 버킷 내 벡터는 PQ 코드(또는 원본 벡터)로 저장되어 있음  
    - 최고 효율의 거리 계산(ADC)으로 Top-K를 빠르게 구함  
  - 필요하면 Top-L 후보를 **re-rank**  
    (원본 벡터 거리 계산 또는 cross-encoder)하여 최종 Top-K 반환  

---

## p50. 벡터 데이터베이스  

<img src="/assets/img/lecture/bigdatasearch/7/image_23.png" alt="image" width="800px"> 

---

## p51. 벡터 데이터베이스: Retrieval  

**Retrieval Algorithms**  
- **유사도 기반 검색**  
- **MMR (Maximal Marginal Relevance) 검색 기법**  
  - 쿼리에 대한 적합한 항목을 검색하면서 동시에 **중복을 최소화** 하는 방법  
  - **관련성과 다양성의 사이의 균형**을 맞추어 검색 결과의 품질을 향상하는 알고리즘  
    - 검색 쿼리에 대한 문서들의 관련성은 최대화  
    - 검색된 문서들 사이의 중복성은 최소화  
      (매우 유사한 항목들만 검색되는 상황 방지)  

  - **동작**  
    - **Greedy** 방식으로 하나씩 선택.  
      각 단계에서 아직 선택되지 않은 문서 중에서  
      다음의 수식의 값을 최대화하는 문서를 선택  

      $$
      MMR = \lambda \cdot Rel(d, Q) - (1 - \lambda) \cdot \max_{d' \in S} Sim(d, d')
      $$

      - $\lambda$: 관련성 vs. 다양성 trade-off 조절 ($0 \le \lambda \le 1$).  
        $\lambda = 1$이면 단순 관련성(다양성 고려 없음),  
        $\lambda = 0$이면 오직 다양성에만 초점.  
      - $Rel(d, Q)$: 쿼리와 문서의 관련도 (예: cosine similarity)  
      - $Sim(d_i, d_j)$: 문서 간 유사도 (예: cosine similarity)  
      - $S$: 이미 선택된 문서 집합  

  >**MMR 선택 절차**  
  >① 초기: $S = \lbrace\rbrace$  
  >② 첫 항목은 보통 가장 관련도 높은 문서(또는 MMR 식으로 선택).  
  >③ 반복: 남은 문서 중 MMR score 최대인 문서를 $S$에 추가.  
  >④ 종료: $|S| = $ 원하는 개수 $K$.  

---

## p52. 벡터 데이터베이스  

**Chroma**  
- **임베딩 벡터를 저장하기 위한 오픈소스 소프트웨어**  
  - 임베딩 및 메타데이터 저장: 임베딩 데이터뿐만 아니라 관련된 메타데이터를 효율적으로 저장  
  - 문서 및 쿼리 임베딩: 텍스트 데이터를 벡터 공간에 매핑하여 임베딩 생성  
  - 임베딩 검색: 사용자 쿼리에 기반하여 가장 관련성이 높은 임베딩을 검색할 수 있음  

- **Chroma 벡터 저장소 생성 & 유사도 기반 검색**  
  - Chroma.from_texts: 분할된 텍스트를 임베딩하고 벡터 저장소에 저장하는 메서드  
  - similarity_search() 메서드  

```python
embeddings_model = OpenAIEmbeddings()
db = Chroma.from_texts(
    texts,                           # chunks로 분할된 텍스트
    embeddings_model,                # 텍스트 임베딩에 사용할 모델
    collection_name='history',       # 저장소 이름
    persist_directory='./db/chromadb',  # 데이터 저장 위치
    collection_metadata={'hnsw:space': 'cosine'}  # l2 is the default, 색인과 검색 알고리즘 지정
)
docs = db.similarity_search(query="누가 한글을 창제했나요?")
```

---

## p53. 벡터 데이터베이스  

```python
loader = TextLoader('history.txt')
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=50,
    encoding_name='cl100k_base'
)
texts = text_splitter.split_text(data[0].page_content)

embeddings_model = OpenAIEmbeddings()
db = Chroma.from_texts(
    texts,
    embeddings_model,
    collection_name='history',
    persist_directory='./db/chromadb',
    collection_metadata={'hnsw:space': 'cosine'}  # l2 is the default
)

query = '누가 한글을 창제했나요?'
docs = db.similarity_search(query)
print(docs[0].page_content)
```

조선은 1392년 이성계에 의해 건국되어, 1910년까지 이어졌습니다.
조선 초기에는 세종대왕이 한글을 창제하여 백성들의 문해율을 높이는 등
문화적, 과학적 성취가 이루어졌습니다. (후략)

---

## p54. 벡터 데이터베이스  

- **Chroma**  
  - **유사도 기반 검색 vs. MMR 검색**

```python
embeddings_model = OpenAIEmbeddings()
db = Chroma.from_texts(
    texts,                          # chunks로 분할된 텍스트
    embeddings_model,               # 텍스트 임베딩에 사용할 모델
    collection_name='history',      # 저장소 이름
    persist_directory='./db/chromadb',  # 데이터 저장 위치
    collection_metadata={'hnsw:space': 'cosine'}  # l2 is the default, 색인과 검색 알고리즘 지정
)

query = '카카오뱅크의 환경목표와 세부추진내용을 알려줘?'

# 유사도 기반 검색
docs = db.similarity_search(query)
print(len(docs))
print(docs[0].page_content)

# MMR(Maximal Marginal Relevance) 기반 검색
mmr_docs = db.max_marginal_relevance_search(query, k=4, fetch_k=10)
print(len(mmr_docs))
print(mmr_docs[0].page_content)
```

---

## p55~56. 벡터 데이터베이스  

- **FAISS**  
  - Facebook AI Research에서 개발  
  - 벡터의 압축된 표현을 사용함으로써 메모리 사용량을 최소화하고 검색 속도를 극대화  

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sbert-nli',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)

vectorstore = FAISS.from_documents(
    documents,
    embedding=embeddings_model,
    distance_strategy=DistanceStrategy.COSINE
)

docs = vectorstore.similarity_search(query)
print(len(docs))
print(docs[0].page_content)

mmr_docs = vectorstore.max_marginal_relevance_search(query, k=4, fetch_k=10)
print(len(mmr_docs))
print(mmr_docs[0].page_content)
```

**MMR 작동 과정**

1. 주어진 쿼리에 대해 유사도가 높은 문서들을 fetch_k 만큼 검색  
2. 해당 문서 집합에서 첫 번째 문서를 선택, 나머지 문서들과의 MMR 점수를 계산  
3. MMR 점수가 가장 높은 문서를 다음으로 선택하고 k개의 문서를 선택할 때까지 반복

---

## p57. 벡터 데이터베이스: Retrieval  

**Retrieval Algorithms**  
- **유사도 기반 검색**  
- **RRF(Reciprocal Rank Fusion): 하이브리드 검색의 최종 순위 생성 알고리즘**  
  - **여러 개의 독립적인 검색기법**의 결과를 결합하여,  
    각 방법의 장점을 살리고 단점을 보완하여 **단일한 최종 순위**를 생성  
  - **하이브리드 검색의 문제점**  
    - 각 검색 방법은 서로 **다른 점수와 순위를 도출하는데**,  
      단순히 점수를 합치면 스케일이 달라 결과가 편향될 가능성이 존재함  
    - RRF는 순위(Rank)를 기반으로 점수의 스케일 문제를 해결  

  - **동작**  
    - 각 검색 방법에서 얻은 순위(r)를 사용하여, 최종적으로 각 문서(d)에 대한 융합 점수(RRFscore)를 계산  
    - $RRFscore(d) = \sum_{i=1}^{N} \frac{1}{k + r_i(d)}$  
      - **RRFscore(d)**: 문서 *d*의 최종 융합 점수  
      - **N**: 사용된 검색 방법의 총 개수 (예: 키워드 검색, 벡터 검색)  
      - **rᵢ(d)**: 검색 방법 *i*에서 문서 *d*가 받은 **순위** (1부터 시작)  
      - **k**: **하이퍼파라미터** (일반적으로 60으로 설정)로 **상위 순위의 영향력**을 조절하는 매개변수

---

## p58. 주요 벡터 DB 비교  

| 구분 | Pinecone | Milvus | Qdrant | Weaviate | Chroma | FAISS | Elasticsearch |
|------|-----------|--------|---------|-----------|---------|--------|----------------|
| 오픈소스 여부 | 상용 (클로즈드 소스) | 오픈소스 (Apache 2.0) | 오픈소스 (Apache 2.0) | 오픈소스 (BSD-3-Clause) | 오픈소스 (MIT) | 오픈소스 (MIT, 라이브러리) | 오픈소스 (Apache 2.0/<br>Elastic License) |
| 핵심 특징 | 완전 관리형,<br>벡터 검색 전용,<br>서버리스 | 클라우드 네이티브<br>분산 아키텍처, 대규모 처리 | Rust 기반,<br>고성능,<br>필터링 & 하이브리드 | 내장 벡터화,<br>GraphQL API,<br>지식 그래프 지향 | Python 네이티브,<br>간편한 로컬 개발 및<br>프로토타이핑 | 순수 벡터 검색<br>라이브러리, GPU 가속 | 기존 검색 엔진,<br>벡터 검색 기능 통합 |
| 하이브리드<br>검색 지원 방법 | Sparse-Dense<br>벡터 통합 (SDA),<br>Metadata 필터링 | Sparse-Dense 통합,<br>Metadata 필터링,<br>RRF(ReciprocalRankFusion) | Vector + Payload 필터<br>(가장 강력),<br>Sparse-Dense 통합 | Vector +<br>Keyword (BM25)<br>통합, RRF | Metadata 필터링<br>(Keyword/Structured) | 외부 시스템 통합<br>필수 (자체 지원 X) | Native BM25 +<br>HNSW 통합 (RRF 사용),<br>가장 성숙한 하이브리드 |
| 주요 인덱싱 지원 | HNSW, IVF | HNSW, IVF_FLAT,<br>ANNOY, DISKANN 등 다양 | HNSW (주력) | HNSW | HNSW (기본) | IVF, PQ, HNSW 등<br>매우 다양 (라이브러리) | HNSW (8.x 버전부터) |
| 확장성 | 최고 수준<br>(완전 관리형,<br>수평 확장 유제한) | 최고 수준 (클라우드 네이티브,<br>분산형 아키텍처) | 매우 좋음<br>(클러스터 및 샤딩 지원) | 좋음 (모듈형 아키텍처,<br>수평 확장 지원) | 낮음<br>(주로 로컬/소규모 POC,<br>엔터프라이즈 기능 부족) | 높음 (대용량 데이터셋<br>처리 가능,<br>라이브러리 레벨) | 좋음 (Elasticsearch<br>클러스터 확장성 활용) |
| QPS (처리량) | 매우 높음<br>(파드 추가로<br>확장 용이),<br>낮은 지연시간 | 매우 높음 (분산 구조),<br>고성능 | 높음 (Rust 기반,<br>메모리 효율) | 준수함 (대규모 시<br>성능 저하<br>가능성 있음) | 낮음 (단일 인스턴스 기준) | 매우 높음 (순수 검색,<br>GPU 가속 시 극대화) | 준수함<br>(통합 검색 부하 고려) |
| 장점 | 운영 오버헤드 無,<br>빠른 구축,<br>엔터프라이즈 기능,<br>SLA 보장 | 유연한 배포(K8s),<br>다양한 인덱스 선택지,<br>대규모 데이터에 최적화 | 고성능 필터링,<br>Rust 기반 안정성과 속도,<br>개발자 친화적 API | 내장 벡터화,<br>객체지향 API 지향,<br>LLM 에코시스템<br>통합 용이 | 가장 쉬운 시작,<br>LLM 프레임워크와의<br>긴밀한 통합, 경량 | 극도의 검색 속도,<br>메모리 효율성,<br>임베딩 모델 무관 | 기존 키워드 검색 통합,<br>성숙한 운영 인프라,<br>강력한 분석 기능 |
| 단점 | 높은 비용,<br>벤더 종속성,<br>커스터마이징 제약 | 초기 설정 및 운영 복잡성<br>학습 곡선 | 대규모 분산 환경에선<br>Milvus 대비 약간 열세 | 데이터 구조 변경 시<br>성능 영향 가능성,<br>재색인 필요 | 대규모 서비스 부족,<br>기본 기능만 제공 | DB 기능 부재(CRUD,<br>Metadata 관리 X),<br>검색 후 데이터<br>재구성 필요 | 순수 벡터 DB 대비<br>성능 최적화가 어려움,<br>리소스 소모가 큼 |
| 주요 활용 사례 | 실시간 개인화,<br>엔터프라이즈<br>RAG 프로젝트,<br>보안 위협 탐지 | 초거대 이미지/비디오 분석,<br>대규모 분산 RAG,<br>계량 분석 | 고성능 추천 시스템,<br>복잡한 메타데이터<br>필터링이 필요한 RAG | 내장형 검색 엔진,<br>지식 그래프 기반<br>Q&A 시스템 | RAG 프로토타입,<br>로컬/테스트 환경의<br>실험 개발 및 실행 | 검색 엔진의 백엔드<br>라이브러리로 임베딩<br>검색 모듈 구현 | 통합 검색 플랫폼<br>(로그, 메타데이터, 벡터),<br>기존 Elastic 사용자 |
| 커뮤니티 규모 | 중간~대형<br>(주로 상용 사용자) | 대형 (가장 활발한 오픈소스<br>커뮤니티 중 하나) | 중간~대형<br>(빠르게 성장 중) | 중간~대형<br>(활발한 개발 및 문서화) | 중간<br>(LLM 개발자에게 인기) | 매우 큼 (AI/ML 분야의<br>표준 라이브러리) | 매우 큼<br>(전통적인 검색 엔진<br>시장 주도) |

---

## p59. Retriever  

**벡터 저장소에서 문서를 검색하는 도구**  
- **Vector Store Retriever**  
  - 대량의 텍스트 데이터에서 관련된 정보를 효율적으로 검색 가능  

- **MultiQueryRetriever**  
  - Vector Store Retriever의 한계를 극복하기 위해 고안된 방식  
  - 쿼리의 의미를 다각도로 포착, 더욱더 관련성이 높고 정확한 정보를 제공  
  - 사용자의 입력 문장을 다양한 관점으로 Paraphrasing함  

- **ContextualCompression**  
  - 검색된 문서 중에서 쿼리와 관련된 정보만을 추출하여 반환  
  - 답변의 품질을 높이고 비용을 줄일 수 있음

---

## p60. Retriever  

**Vector Store Retriever**  
- **사전 준비:** 문서 임베딩을 벡터스토어에 저장  

- **단일 문서 검색**  
  → 유사도가 가장 높은 것  

- **MMR 검색**  
  → lambda_mult: 유사도와 다양성 중요도 결정  

- **유사도 점수 임계값 기반 검색**  
  → 유사도가 임계값 이상인 문서만을 대상으로 추출  

- **메타데이터 필터링 사용한 검색**  
  - 메타데이터의 특정 필드에 대해서 기준을 설정하여 필터링  
  - 예: 'format': 'PDF 1.4'처럼 특정 형식이나 조건 만족 기준

---

## p61. Retriever  

**Multi Query Retriever**  

- **사용 방법**  
  1. **MultiQueryRetriever 설정**  
     - `from_llm`을 통해 기존 벡터 저장소 검색도구와 LLM 모델을 결합하여 인스턴스 생성  
  2. **로깅 설정**  
     - 생성되고 실행되는 쿼리들에 대한 정보를 로그로 기록하고 확인 가능  
  3. **문서 검색 실행**  
     - `get_relevant_documents` 메서드를 사용하여 쿼리에 대한 Multi Query 기반의 문서 검색 진행  
  4. **결과 확인**  
     - 검색을 통해 반환된 고유 문서들의 수를 확인

---

## p62. Retriever  

**Contextual compression**  

- **구성과 동작방식**  
  1. **LLMChainExtractor 설정**  
     - `LLMChainExtractor.from_llm(llm)`을 사용하여 문서 압축기를 설정  
     - 언어모델을 사용하여 문서 내용을 압축 (질의에 관련된 내용만 추출)  

  2. **ContextualCompressionRetriever 설정**  
     - `base_compressor`에 LLMChainExtractor 인스턴스를,  
       `base_retriever`에 기본 검색기 인스턴스를 제공  

  3. **압축된 문서 검색**  
     - `Compression_retriever.get_relevant_documents(question)` 함수를 사용하여  
       주어진 쿼리에 대한 압축된 문서들을 검색  
     - 기본 검색기를 통해 얻은 문서들은 문서 압축기를 사용하여  
       내용을 압축 및 가장 관련된 내용만 추려냄  

  4. **결과 출력**

---

## p63. Generation  

**Generation의 단계**  
1. **Retrieval**  
    - MRR을 사용하든, 일반적인 similarity search를 사용하든 문서를 검색  
2. **Prompt**  
    - Context 부분에 문서를 넣어줌  
3. **Model**  
    - Model을 초기화  
4. **Formatting Docs**  
    - 검색된 문서를 포맷팅  
5. **Chain Execution**  
    - LLM chain 구성 및 실행  
6. **Run**  
    - Invoke를 사용하여 체인을 실행

---

## p64. Generation  

```python
# Retrieval
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'lambda_mult': 0.15}
)

docs = retriever.get_relevant_documents(query)

# Prompt
template = '''Answer the question based only on the following context:
{context}

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

# Model
llm = ChatOpenAI(
    model='gpt-3.5-turbo-0125',
    temperature=0,
    max_tokens=500,
)

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

# Chain
chain = prompt | llm | StrOutputParser()

# Run
response = chain.invoke({'context': (format_docs(docs)), 'question': query})
response
```

```
카카오뱅크의 환경목표는 
‘더 나은 세상을 만들어 나가는데 앞장서겠습니다’입니다.  

이를 위해 카카오뱅크는 다음과 같은 환경방침을 수립하여 운영하고 있습니다:  
- 전사 환경경영 정책 수립  
- 녹색 구매 지침 수립  
- 환경 지표 설정 및 성과 관리  
- 자원 사용량 관리 (용수, 폐기물, 에너지 등)  
- 기후변화를 포함한 환경 리스크 관리체계 마련  
- Scope 1&2&3 온실가스 배출량 모니터링  
- 탄소 가격 도입을 통한 환경 비용 관리  
- 신재생 에너지 사용 확대  
- 녹색채권 발행 기반 마련  
- 환경경영 조직 및 관리 체계 구축  
- 환경영향평가 체계 구축  
- 구축을 통한 환경 리스크 및 성과 관리  

이를 통해 카카오뱅크는 환경을 보호하고  
지속 가능한 경영을 추구하고 있습니다.
```

---

## p65. Legal Query RAG(LQ-RAG)’ 프레임워크 (출처 : 인공지능신문)

<img src="/assets/img/lecture/bigdatasearch/7/image_24.png" alt="image" width="800px"> 

---

### 보충 설명  

#### 1. **Legal Query RAG(LQ-RAG)의 개요**  
- LQ-RAG는 법률 질의(legal query)에 특화된 **RAG(Retrieval-Augmented Generation)** 프레임워크이다.  
- 일반적인 RAG 구조를 기반으로 하되, **법률 문서의 복잡성과 정확성 요구**를 충족하기 위해  
  **임베딩 LLM(Embedding LLM)** 과 **생성 LLM(Generative LLM)** 을 각각 정교하게 조정(fine-tuning)한다.  
- 사용자의 질의(Query)를 받아 관련 법률 데이터에서 근거를 검색하고,  
  최종적으로 신뢰할 수 있는 답변을 생성한다.  

#### 2. **RAG Layer (상단 구조)**  
- **사용자 질의(User Query)** 는 임베딩 LLM을 통해 **벡터 표현(Dense Representation)** 으로 변환된다.  
- 변환된 쿼리는 **Vector DB** 로 전달되어, 유사한 법률 문서의 **Top-K Retrieval** 이 수행된다.  
- 검색된 문서 조각(Chunks)은 **Re-ranker** 를 통해 재정렬되어,  
  품질이 높은 근거 문서가 **Generative LLM** 으로 전달된다.  
- 이후 생성된 초안 답변(Initial Response)은 **Evaluation Agent** 가 검증하며,  
  필요 시 **Feedback Loop** 을 통해 프롬프트나 검색 과정을 다시 조정한다.  
- 최종적으로 품질이 확보된 답변만 **Final Response** 로 사용자에게 제공된다.  

#### 3. **Fine-Tuning Layer (하단 구조)**  
LQ-RAG는 두 가지 세부 파인튜닝 과정으로 구성된다.  

1. **Embedding LLM Fine-Tuning Process**  
   - 비정형 법률 데이터(Unstructured Legal Data)를 수집하고 전처리하여  
     일반 임베딩 모델(General Purpose Embedding Model)에 학습시킨다.  
   - **LLM 기반 데이터 생성(Data Generator)** 과 **피드백 루프(Feedback Loop)** 를 통해  
     법률 도메인에 최적화된 **Fine-Tuned Embedding LLM** 을 완성한다.  
   - 이렇게 만들어진 임베딩 모델은 RAG 단계에서 **정확한 법률 문서 검색**을 수행한다.  

2. **Generative LLM Fine-Tuning Process**  
   - 질의응답(QA) 및 지시문(Instructions) 형태의 데이터셋을 활용한다.  
   - **PEFT(매개변수 효율적 파인튜닝)** 와 **LoRA(Low-Rank Adaptation)** 를 적용해  
     기존 LLM의 일부만 효율적으로 조정한다.  
   - 이후 병합(Merging)과 평가(Model Evaluation)를 거쳐  
     법률 질의에 특화된 **Fine-Tuned Generative LLM** 을 구축한다.  

#### 4. **핵심 특징 요약**  
- **법률 특화 쿼리 임베딩**: 일반적인 벡터 검색보다 정확한 법률 용어 매칭 가능.  
- **ReAct Agent 기반 하이브리드 검색**: 질의를 다중 관점으로 변환(EQ)하여 검색 품질 향상.  
- **Feedback Loop을 통한 품질 개선**: RAG와 파인튜닝 양쪽 모두에서 반복적 개선 구조 채택.  
- **최종 목표**: 정확하고 근거 기반의 **법률 질의 응답 시스템(Legal QA System)** 구현.  

> 출처: 인공지능신문 (https://www.aitimes.kr)