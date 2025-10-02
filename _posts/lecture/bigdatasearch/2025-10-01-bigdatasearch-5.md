---
layout: post
title: "[빅데이터와 정보검색] 5주차 LLM의 이해와 프롬프트 엔지니어링"
date: 2025-10-01 22:00:00 +0900
categories:
  - "대학원 수업"
  - "빅데이터와 정보검색"
tags: []
---

> 출처: 빅데이터와 정보검색 – 황영숙 교수님, 고려대학교 (2025)

## p2. 강의 개요

거대 언어 모델(Large Language Model, LLM)은 폭발적인 성장을 거듦…  
단순히 사용자의 간단한 질문에 답변하는 어시스턴트 수준을 넘어,  
최근에는 **모델 스스로 생각하고 추론하는 능력까지 갖추어**….

이러한 추론 능력의 발전과 함께  
검색, 컴퓨터 조작, API 호출이나 코드 실행과 같은 다양한 도구를 활용하여  
**사용자를 대신해 복잡한 작업을 수행하고, 편의성을 크게 향상**

- 주제  
  - LLM의 구조 이해 및 프롬프트 엔지니어링 학습  

- 목표  
  - LLM과 트랜스포머 모델의 핵심 이해  
  - 검색 관련 다양한 프롬프트 기법 습득 및 활용  

---

## p3. LLM의 기본개념

- LLM(Large Language Model)  
  - 대규모 언어 데이터로 언어의 구조와 의미를 학습한 매우 큰 규모의 인공지능 언어모델  
  - 인간의 언어를 처리하고 맥락을 이해하며 텍스트 생성, 번역, 요약, 질문에 대한 답변 등 다양한 언어 관련 작업을 수행하는 모델  

- 언어모델(Language Model)?  
  - 언어 모델은 언어 현상을 모델링하고자 **단어 시퀀스(문장)에 확률을 할당**하는 모델  
  - 언어 모델은 가장 자연스러운 **단어 시퀀스를 찾아내어 생성**하는 모델  
  - 언어를 이루는 구성 요소(글자, 형태소, 단어, 단어열 혹은 문장, 문단 등)를 문맥으로 하여  
    **다음 구성 요소를 예측/생성하는 모델**  

---

## p4. LLM의 기본개념

- **단어 시퀀스 확률**  
  - 하나의 단어를 w, 단어 시퀀스를 W라고 한다면, n개의 단어가 등장하는 단어 시퀀스 W의 확률은  

    $$
    P(W) = P(w_1, w_2, w_3, w_4, w_5, \ldots, w_n)
    $$

  - 다음 단어의 등장 확률 : n-1개의 단어가 나열된 상태에서 n번째 단어의 확률은  

    $$
    P(w_n \mid w_1, \ldots, w_{n-1})
    $$

  - 전체 단어 시퀀스 W의 확률은  

    $$
    P(W) = P(w_1, w_2, w_3, w_4, w_5, \ldots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, \ldots, w_{i-1})
    $$  

    $$
    P(w_1, w_2, w_3, \ldots, w_n) = P(w_1) P(w_2 \mid w_1) P(w_3 \mid w_1, w_2) \cdots P(w_n \mid w_1, \ldots, w_{n-1})
    $$  

    $$
    = \prod_{n=1}^{n} P(w_n \mid w_1, \ldots, w_{n-1})
    $$

---

### 보충 설명  

#### 1. **단어 시퀀스 확률의 의미**  
- 문장은 단어들의 시퀀스로 표현할 수 있으며,  
  전체 문장의 확률은 각 단어가 이전 단어들 뒤에 올 조건부 확률의 곱으로 정의된다.  

$$
P(W) = P(w_1, w_2, \ldots, w_n) = \prod_{i=1}^n P(w_i \mid w_1, \ldots, w_{i-1})
$$  

#### 2. **카운팅 기반 확률 추정**  
- 전통적인 언어모델에서는 확률을 **단어 등장 빈도**로 계산했다.  
- 특정 단어 $w_n$이 앞선 시퀀스 $(w_1, \ldots, w_{n-1})$ 뒤에 올 확률은  

$$
P(w_n \mid w_1, \ldots, w_{n-1}) = \frac{\text{Count}(w_1, \ldots, w_{n-1}, w_n)}{\text{Count}(w_1, \ldots, w_{n-1})}
$$  

#### 3. **희소성(Sparsity) 문제**  
- 실제 언어 데이터에서는 특정 시퀀스가 거의 등장하지 않기 때문에  
  위와 같은 확률이 0이 되는 경우가 많다.  
- 이러한 희소성 문제를 해결하기 위해 스무딩 기법이나 근사화 방법이 사용된다.  

#### 4. **N-그램 근사와 단순화**  
- 모든 과거 단어를 고려하면 계산량이 지나치게 커지므로,  
  보통 최근 N-1개의 단어만 조건으로 사용하는 **N-그램 모델**이 활용된다.  

$$
\text{Bigram: } P(w_n \mid w_{n-1}), \quad \text{Trigram: } P(w_n \mid w_{n-2}, w_{n-1})
$$  

- 엄밀하게는 Bigram에서의 확률을 구해야 하지만 계산량이 크기 때문에,  
  다음과 같이 근사하기도 한다:  

$$
P(w_2 \mid w_1) \approx \frac{P(w_1 \cap w_2)}{P(w_1)}
$$  

- 이러한 근사화는 실제 계산을 단순화하면서도 일정 수준의 성능을 보장한다.  

---

## p5. 언어모델의 발전과정

<img src="/assets/img/bigdatasearch/5/image_1.png" alt="image" width="720px">

- **특정 작업 보조기 (Specific task helper)**  
  - **n-그램 모델 (n-gram models)**  
  - 통계적 방법 (Statistical methods)  
  - 확률 추정 (Probability estimation)  
  - 특정 작업 보조 (*Assist in specific tasks*)  
  - → 통계적 언어모델 (Statistical LM)  

- **작업 비의존적 특징 학습기 (Task-agnostic feature learner)**  
  - **Word2vec (NPLM), NLPS**  
  - 정적 단어 표현 (Static word representations)  
  - 신경망 기반 맥락 모델링 (Neural context modeling)  
  - 전형적인 NLP 작업 해결 (*Solve typical NLP tasks*)  
  - → 신경망 언어모델 (Neural LM)  

- **전이 가능한 NLP 작업 해결기 (Transferable NLP task solver)**  
  - **ELMO, BERT, GPT-1/2**  
  - 맥락 인식 표현 (Context-aware representations)  
  - 사전 학습 + 미세조정 (Pre-training + fine-tuning)  
  - 다양한 NLP 작업 해결 (*Solve various NLP tasks*)  
  - → 사전학습 언어모델 (Pre-trained LM)  

- **범용 작업 해결기 (General-purpose task solver)**  
  - **GPT-3/4, ChatGPT, Claude**  
  - 대규모 언어모델 (Scaling language models)  
  - 프롬프트 기반 완성 (Prompt based completion)  
  - 다양한 실제 작업 해결 (*Solve various real-world tasks*)  
  - → 대규모 언어모델 (LLM)  

- **연대표**  
  - 1990년대: 마르코프 가정 기반 **n-그램 언어모델 (n-gram LM)**  
  - 2013년: 전형적 NLP 처리 (NLM, Word2Vec, Seq2Seq, 어텐션 매커니즘 발전)  
  - 2018년: **트랜스포머 기반 언어모델 (Transformer-based LM)**  
    - “사전 학습 및 미세조정(Pre-training & Fine-tuning)” 학습 패러다임  
  - 2020년: **대규모 PLM(Pre-trained Language Model, 모델 크기/데이터)**  
    - 놀라운 문제 해결 능력 발견  

---

### 보충 설명  

#### 1. **통계적 언어모델 (Statistical LM, n-gram)**  
- 1990년대의 전형적 언어모델로, **마르코프 가정**에 기반하여 직전 n-1개의 단어를 보고 다음 단어를 예측한다.  
- 주로 **문자/단어 예측, 음성 통사 태깅** 등에 활용되었으며, 특정 Task만 제한적으로 수행할 수 있었다.  
- “바로 앞의 특정한 개수의 단어에만 의존”하는 한계가 있어 **긴 맥락 반영이 어렵다**.  

#### 2. **신경망 언어모델 (Neural LM)**  
- 2013년경 **Word2Vec, Seq2Seq, RNN, Attention 메커니즘**이 등장하며 단어를 **벡터로 임베딩**하고 문맥 정보를 학습하기 시작했다.  
- 정적 단어 표현(Word2Vec)과 함께 문맥 기반 표현(Neural Context Modeling)을 활용하여 **Task-agnostic feature learner**로 발전했다.  
- “Task 다양화”가 가능해졌으나, 여전히 구글 번역 등 특정 애플리케이션에 한정되는 경우가 많았다.  

#### 3. **사전학습 언어모델 (Pre-trained LM)**  
- 2018년 **Transformer** 기반 모델(BERT, GPT-1/2, ELMo 등)이 등장하면서 **사전 학습 + 미세조정(fine-tuning)** 패러다임이 확립되었다.  
- 이 방식은 하나의 거대한 언어모델을 학습시킨 뒤, 각 작업에 맞게 조금씩 조정하여 다양한 NLP 작업에 적용할 수 있도록 했다.  
- 구글이 BERT를 공개하며 NLP 연구에 큰 변화를 이끌었다.  

#### 4. **대규모 언어모델 (LLM)**  
- 2020년 이후 **GPT-3/4, ChatGPT, Claude**와 같은 초대형 모델이 등장하면서,  
  **범용 작업 해결기(General-purpose task solver)**로 진화하였다.  
- 단순한 NLP 작업을 넘어 **실제 문제 해결 능력(emergent ability)**이 부각되었으며,  
  프롬프트를 기반으로 검색, 요약, 코드 작성, 번역 등 다양한 실세계 작업을 수행할 수 있게 되었다.  
- **데이터와 모델 크기의 폭발적 증가**가 이러한 능력 향상의 핵심 요인이었다.  

#### 5. **정리**  
- **n-gram → Neural LM → Pre-trained LM → LLM**의 발전 과정은  
  “국소적 예측 → 맥락 반영 → 전이학습 → 범용 지능”으로 확장된 역사이다.  
- 각 단계는 전 단계의 한계를 극복하며 새로운 패러다임을 제시하였고,  
  특히 LLM은 **검색·코드 실행·복잡한 의사결정까지 확장된 응용 가능성**을 보여준다.  

---

## p6. 트랜스포머 모델의 이해와 활용

<img src="/assets/img/bigdatasearch/5/image_2.png" alt="image" width="360px">

- **병렬처리와 모델 확장 가능한 트랜스포머**  
  - Encoder와 Decoder 연결 구조  
  - Multi-Head Attention Mechanism으로 병렬처리와 장기 의존성 문제 해결  
  - 다수의 Encoder 층과 Decoder 층 연결로 모델 용량 확장  
  - Auto-Regressive Decoding Model  
  - 최종 비용/손실함수는 softmax 분류기 함수에 대한 교차 엔트로피 (Cross Entropy)를 사용  

- **언어모델 매개변수 확장(수백억 또는 수천억 개) → 다양한 LLM 개발의 백본!**

---

### 보충 설명  

#### 1. **Encoder-Decoder 구조**  
- 트랜스포머는 **Seq-to-Seq(Sequence-to-Sequence)** 모델의 확장으로, 입력(소스 언어)을 인코더가 처리하고 출력(타겟 언어)을 디코더가 생성한다.  
- 인코더는 입력 문장의 맥락(Context Vector)을 벡터로 압축하고, 디코더는 이를 참조해 출력 문장을 점진적으로 생성한다.  
- “source → target” 구조가 명확하다.  

#### 2. **Multi-Head Attention의 역할**  
- RNN의 한계였던 **장기 의존성 문제**를 해결하기 위해 도입되었다.  
- 여러 개의 Attention Head를 병렬로 두어 문맥을 다양한 관점에서 해석할 수 있다.  
- 이는 병렬 연산이 가능하게 하여 학습 속도를 크게 향상시킨다.  

#### 3. **모델 용량 확장 (스택 구조)**  
- 인코더와 디코더는 단일 블록이 아니라 **동일한 층을 여러 번 쌓아 구성**된다(논문 기준 6층).  
- 층이 깊어질수록 더 복잡한 문맥과 의미를 학습할 수 있다.  
- “모델 용량 확장”은 결국 **매개변수(parameter) 수 증가**와 직결되며, 이는 LLM 발전의 핵심이다.  

#### 4. **Feed Forward와 비선형 변환**  
- 각 Attention 블록 뒤에는 **비선형 변환(Non-linear projection)** 이 포함된 Feed Forward 네트워크가 있다.  
- 이 구조는 단순한 선형 변환으로는 잡히지 않는 복잡한 패턴을 학습하는 데 필수적이다.  

#### 5. **Cross Attention과 Auto-Regressive Decoding**  
- 디코더는 자체 입력(이전 단어) + 인코더 출력(Context Vector)을 함께 참조한다.  
- 이때 인코더의 출력이 디코더 Attention에 연결되는 부분을 **Cross Attention**이라 한다.  
- 출력은 Auto-Regressive 방식으로 한 단어씩 순차적으로 생성된다.  

#### 6. **언어모델 매개변수 확장의 의미**  
- 수백억~수천억 개의 매개변수를 가진 모델로 확장되면서,  
  트랜스포머는 단순 번역이나 요약을 넘어 **범용 LLM 개발의 백본(backbone)**이 되었다.  
- “Source language → Target language” 구조를 기반으로 다양한 작업에 적용할 수 있는 토대가 마련된 것이다.  

---

## p7. 트랜스포머 모델

**동작 과정**

<img src="/assets/img/bigdatasearch/5/image_3.png" alt="image" width="720px">

---

### 보충 설명  

#### 1. **트랜스포머 인코더(Encoder)의 역할**  
- 입력 문장(예: *Je suis étudiant*)은 먼저 **임베딩(embedding)** 과 **위치 정보(positional encoding)** 가 더해져 인코더로 전달된다.  
- 인코더는 입력 문장의 모든 단어를 동시에 처리하며, **문맥(Context) 정보를 풍부하게 담은 표현 벡터**를 생성한다.  
- 이렇게 얻은 표현은 이후 디코더가 참조할 수 있도록 **Key(K)** 와 **Value(V)** 벡터로 변환된다.  

#### 2. **트랜스포머 디코더(Decoder)의 역할**  
- 디코더는 이미 생성된 단어(Previous outputs)를 참고하면서 다음 단어를 예측한다.  
- 인코더에서 전달된 **K, V 벡터**와 디코더 내부의 **Query(Q)** 를 통해 **어텐션(Attention)** 연산을 수행한다.  
- 이를 통해 입력 문장의 중요한 부분에 집중하면서 출력 단어를 순차적으로 생성한다.  

#### 3. **Auto-Regressive 방식과 Softmax 출력**  
- 디코더는 한 번에 전체 문장을 내놓지 않고, **자동 회귀(Auto-regressive)** 방식으로 단어를 하나씩 생성한다.  
- 예를 들어, 현재 “Je suis étudiant”라는 입력에 대해, 첫 출력이 “I”라면, 다음 출력은 이 단어를 포함한 맥락을 바탕으로 예측한다.  
- 최종적으로 디코더의 출력은 **Linear 층**을 거쳐 **Softmax 분류기**에서 확률 분포로 변환되어 가장 적합한 단어가 선택된다.  

#### 4. **핵심 개념 요약**  
- 인코더는 **입력 문장의 맥락을 벡터로 압축**하고,  
- 디코더는 이를 바탕으로 **출력 문장을 순차적으로 생성**한다.  
- 이 과정은 번역, 요약, 질의응답 등 다양한 NLP 태스크에서 공통적으로 적용된다.  

---

## p8. 트랜스포머 모델

<img src="/assets/img/bigdatasearch/5/image_2.png" alt="image" width="360px">

**모델의 구조**  

- 입력 임베딩 (Input Embedding)  
  - 입력 시퀀스(문장)의 각 단어를 벡터로 변환  

- 포지셔널 인코딩 (Positional Encoding)  
  - 단어의 순서 정보를 벡터에 추가  
  - 트랜스포머는 순환 구조가 없으므로, 단어의 위치 정보를 별도로 주입함.  

- 인코더 (Encoder)  
  - 입력 시퀀스의 정보를 압축하여 문맥 정보를 담은 벡터(Context Vector) 생성  
  - 여러 개의 동일한 인코더 레이어를 쌓아서 구성(논문에서는 6개)  

- **인코더의 최종 출력은 각 디코더 레이어의 Encoder-Decoder Attention 블록으로 연결**  

- 디코더 (Decoder)  
  - 인코더의 출력(Context Vector)과 이전 스텝에서 생성된 출력 단어를 입력받아 다음 단어를 예측  
  - 여러 개의 동일한 디코더 레이어를 쌓아서 구성(논문에서는 6개).  

- 선형 변환 및 소프트맥스 (Linear & Softmax)  
  - 디코더의 최종 출력 벡터를 기반으로 다음 단어의 확률 분포를 계산  

---

### 보충 설명  

#### 1. **입력 임베딩과 위치 정보**  
- 트랜스포머는 순환 구조(RNN)가 없기 때문에 단어의 순서를 직접적으로 알 수 없다.  
- 이를 해결하기 위해 각 단어를 **벡터로 임베딩**한 후, 추가적으로 **포지셔널 인코딩(Positional Encoding)**을 더해 단어의 순서 정보를 반영한다.  
- 이 과정을 통해 모델은 단어의 의미뿐만 아니라 문장 내 위치까지 학습할 수 있다.  

#### 2. **인코더의 역할**  
- 인코더는 입력 시퀀스 전체를 받아 문맥을 담은 **컨텍스트 벡터(Context Vector)**를 생성한다.  
- 여러 층의 인코더 레이어를 쌓아 깊은 문맥 표현을 학습하며, 각 층에는 **멀티헤드 어텐션**과 **피드포워드 네트워크**가 포함된다.  
- 인코더의 최종 출력은 디코더와 연결되어 번역, 요약 등의 작업에서 핵심적인 단서가 된다.  

#### 3. **디코더의 역할**  
- 디코더는 인코더의 출력(Context Vector)과 이전에 생성된 단어를 입력으로 받아 **다음 단어를 예측**한다.  
- 마스크드 어텐션(Masked Attention)을 사용하여, 아직 생성되지 않은 미래 단어는 참조하지 못하도록 한다.  
- 이렇게 함으로써 **자동회귀(Auto-Regressive) 방식**으로 문장을 한 단어씩 순차적으로 생성한다.  

#### 4. **출력과 확률 분포**  
- 디코더의 최종 출력은 **선형 변환(Linear)**을 거친 뒤 **소프트맥스(Softmax)**에 입력되어,  
  가능한 모든 단어에 대한 확률 분포로 변환된다.  
- 이 확률 분포에서 가장 높은 값을 가진 단어가 실제 출력으로 선택된다.  

#### 5. **전체 구조의 의미**  
- 트랜스포머는 **입력 임베딩 + 위치 정보 → 인코더 → 디코더 → 확률 분포**라는 구조로,  
  병렬처리와 장기 의존성 학습이 가능하다.  
- 이 아키텍처는 번역, 질의응답, 요약, 텍스트 생성 등 현대 LLM의 기본 골격을 형성한다.  

---

## p9. 트랜스포머 모델

- **Input Embedding & Positional Encoding**  
  - 텍스트 시퀀스는 Tokenizer를 통해 토큰으로 분리 후, Vocab Dictionary 내 토큰의 인덱스 ID로 표현하고,  
  - 각 토큰은 임베딩 모델을 사용하여 고정크기($d_{model}$)의 벡터로 임베딩 (논문에서는 $d_{model}=512$)  
  - 각 토큰의 임베딩 벡터와 토큰의 위치를 인코딩한 벡터를 더하여 입력 벡터를 구성  

<img src="/assets/img/bigdatasearch/5/image_4.png" alt="image" width="720px">

---

### 보충 설명  

#### 1. **토큰화와 임베딩**  
- 입력 문장은 먼저 **Tokenizer**를 통해 토큰 단위로 분리된다.  
- 각 토큰은 **Vocab Dictionary**에 의해 정수 ID로 변환되고,  
  이 ID는 다시 **임베딩 벡터**로 매핑된다.  
- 임베딩 벡터의 차원은 고정 크기 $d_{model}$이며, 논문에서는 $d_{model} = 512$로 설정되었다.  

#### 2. **포지셔널 인코딩(Positional Encoding)**  
- 트랜스포머는 RNN과 달리 순차적 구조가 없기 때문에 **단어의 위치 정보**를 직접 주입해야 한다.  
- 위치 인코딩은 사인(sin), 코사인(cos) 함수를 이용해 각 위치에 대한 주기적 패턴을 만들어낸다.  

$$
PE_{(pos, 2i)} = \sin \left( \frac{pos}{10000^{2i/d_{model}}} \right)
$$  

$$
PE_{(pos, 2i+1)} = \cos \left( \frac{pos}{10000^{2i/d_{model}}} \right)
$$  

- 여기서 $pos$는 시퀀스 내 토큰의 위치, $i$는 벡터 내 차원의 인덱스를 의미한다.  

#### 3. **입력 벡터 구성**  
- 최종 입력 벡터는 **임베딩 벡터 + 포지셔널 인코딩 벡터**의 합으로 계산된다.  
- 이렇게 구성된 입력은 이후 인코더로 전달되어 문맥적 의미를 학습할 수 있게 된다.  

#### 4. **핵심 의의**  
- 임베딩은 단어의 **의미 표현**,  
- 포지셔널 인코딩은 단어의 **순서 정보**를 담아,  
  두 요소가 결합함으로써 문장의 **구조적 의미**가 모델에 전달된다.  

---

## p10. 트랜스포머 모델

- **Positional encoding의 조건**  
  - 각 Time Step (문장 내 단어의 위치) 에 대해 고유한 인코딩을 출력해야 함  
  - 두 Time Step 사이의 거리는 문장의 길이와 상관없이 일관되어야 함  
  - 값들은 문장의 길이와 상관없이 Bounded 되어야 함  

$$
PE_{(pos, 2i)} = \sin \left( \frac{pos}{10000^{2i/d_{model}}} \right),
$$  

$$
PE_{(pos, 2i+1)} = \cos \left( \frac{pos}{10000^{2i/d_{model}}} \right)
$$  

- 논문에서의 설정:  
  $d_{model} = 512 \quad (i.e., i \in [0, 255])$  

<img src="/assets/img/bigdatasearch/5/image_5.png" alt="image" width="720px">

---

### 보충 설명  

#### 1. **Positional Encoding의 필요성**  
- 트랜스포머는 순환 구조(RNN)나 CNN처럼 위치 정보를 직접 반영하지 않는다.  
- 따라서 각 **단어의 순서 정보**를 모델에 전달하기 위해 **Positional Encoding**이 사용된다.  
- 이 인코딩은 **sin, cos 함수**를 활용하여 위치에 따른 주기적 패턴을 만들어내며, 단어의 상대적 순서를 학습할 수 있도록 한다.  

#### 2. **Positional Encoding의 조건**  
- 각 Time Step(문장 내 단어의 위치)에 대해 **고유한 인코딩**을 출력해야 한다.  
- 두 Time Step 사이의 거리는 문장의 길이와 무관하게 일정해야 한다.  
- 인코딩 값들은 문장의 길이에 상관없이 **Bounded(제한된 범위)** 내에 있어야 한다.  

#### 3. **구체적인 정의**  
- 위치 $pos$, 차원 인덱스 $i$, 임베딩 크기 $d_{model}$ 에 대해 정의는 다음과 같다.  

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$  

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$  

- 논문에서는 $d_{model} = 512$ 로 설정되었으며, $i \in [0, 255]$ 범위의 인덱스를 가진다.  

#### 4. **로터리 포지셔닝 (Rotary Position Embedding, RoPE)**  
- 기존의 sin/cos 기반 인코딩을 확장한 방법으로, **내적 연산**에 직접 회전 행렬을 적용하여 위치 정보를 반영한다.  
- 장점:  
  - **상대적 위치(Relative Position)**를 자연스럽게 반영할 수 있다.  
  - 매우 긴 시퀀스에서도 안정적인 학습이 가능하다.  
- RoPE는 최근 LLM(예: LLaMA 계열)에서 널리 사용되며, 기존 절대적 위치 인코딩보다 확장성이 뛰어나다.  

#### 5. **긴 Context 처리와 RAG에서의 역할**  
- RAG(Retrieval-Augmented Generation)처럼 긴 문맥을 다루는 시스템에서는 **수천~수만 토큰**을 처리해야 한다.  
- 이때 단순한 포지셔널 인코딩은 한계를 보이므로, **더 정교한 위치 표현 기법**이 필요하다.  
- Positional Encoding(특히 RoPE와 같은 방식)은 긴 context window에서도 **위치 정보를 안정적으로 유지**하여,  
  검색된 문서와 입력 쿼리 간의 정렬(alignment)을 가능하게 한다.  

#### 6. **의의**  
- Positional Encoding은 단순한 "위치 태그"가 아니라,  
  **순서와 간격을 연속적으로 반영**하여 모델이 긴 문맥 속에서도 정확히 위치를 구분할 수 있게 한다.  
- 이는 현대 LLM이 **긴 컨텍스트 이해, 검색 증강(RAG), 대화 지속성 유지**와 같은 기능을 수행하는 데 핵심적인 역할을 한다.  

---

## p11. 트랜스포머 모델

- **Transformer의 Attention**  
  - Encoder의 Self-Attention  
  - Decoder의 Masked Self-Attention  
  - Decoder의 Encoder-Decoder Attention  

- **Attention?**  
  - 입력 문장에서 **중요한 부분에 집중**하는 메커니즘  
  - 입력문장의 각 단어 간 연관성을 수치화하고, 문장 내에서 중요한 단어들에게 가중치를 부여  

- **Self-Attention의 장점**  
  - 문장 내 단어 간 관계를 계산하여 의미적 연관성을 반영  
  - 문장 내 **모든 단어 간 관계를 한 번에 학습**  
  - **병렬 연산 가능** → 연산 속도 향상  
  - 긴 문장에서도 효율적으로 문맥을 유지  

<img src="/assets/img/bigdatasearch/5/image_6.png" alt="image" width="600px">

---

## p12. 트랜스포머 모델

**Multi-Head Attention Block의 내부 구조**  

<img src="/assets/img/bigdatasearch/5/image_7.png" alt="image" width="720px">

---

### 보충 설명  

#### 1. **Scaled Dot-Product Attention의 핵심 원리**  
- Query(Q)와 Key(K)의 내적을 통해 단어 간 유사도를 측정한다.  
- 차원이 커질수록 내적 값이 커지므로, 안정적인 학습을 위해 $\sqrt{d_k}$로 나누어 스케일링한다.  
- 필요할 경우 마스킹(Mask)을 적용하여 미래 단어를 보지 못하게 하거나 패딩 토큰의 영향을 제거한다.  
- Softmax를 통해 확률 분포 형태의 가중치를 계산한 뒤, Value(V)에 곱하여 최종 Attention 출력을 얻는다.  

#### 2. **Multi-Head Attention의 동작 방식**  
- Q, K, V를 여러 개의 head로 분리하여 병렬적으로 Attention 연산을 수행한다.  
- 각 head는 입력의 다른 표현 공간을 학습하기 때문에 다양한 시각에서 문맥을 이해할 수 있다.  
- 여러 head의 출력을 Concatenate한 뒤, Linear 변환을 거쳐 최종 출력 벡터를 생성한다.  

#### 3. **구조적 특징과 장점**  
- **다양한 관계 학습**: 단일 Attention보다 다양한 의미적 연관성을 동시에 학습할 수 있다.  
- **정교한 문맥 반영**: 긴 문장에서 단어 간의 복잡한 의존성을 포착하고, 문맥을 더 풍부하게 표현한다.  
- **병렬 연산 가능**: 여러 head가 동시에 동작하므로 계산 효율성이 높고, 학습 속도가 향상된다.  

---

## p13. 트랜스포머 모델

- **Multi-Head Self-Attention**  
  - Self-Attention은 입력 시퀀스의 모든 단어를 다른 모든 단어와 연결하여 상호 연관성을 구하는데,  
  - Scaled Dot-Product Attention 알고리즘 적용하여 단어간 Attention Score를 계산하되  
  - Query, Key, Value 벡터를 h개 Head로 분할하여 h개의 AttentionScore를 구한 후 결합(Concat)하여 어텐션 가중치를 구한다. 그리고 어텐션 가중치를 Value 벡터에 적용하여 최종 Attention Value 벡터를 도출함  

<img src="/assets/img/bigdatasearch/5/image_8.png" alt="image" width="600px">

- **Q(Query)** : 입력시퀀스에서 관련된 부분을 찾으려고 하는 소스 벡터  
- **K(Key)** : 관계의 연관도를 찾기 위해 쿼리와 비교하는 대상 벡터  
- **V(Value)** : 특정 Key에 해당하는 입력 시퀀스의 정보로 가중치를 구하는데 사용  

<img src="/assets/img/bigdatasearch/5/image_9.png" alt="image" width="360px">

- 입력 문장 내의 단어들끼리 상호 연관성을 구함으로써 “making”이 “difficult”과 가장 연관이 많이 되고 있음을 알아냄  

---

## p14. 트랜스포머 모델

- **Multi-Head Attention ?**  
  - 단일 Attention만 사용하면 하나의 관계만 학습  
  - 여러 개의 Attention Head를 사용하여 다양한 관계 및 언어 현상/패턴을 학습  
  - h개의 서로 다른 Attention을 적용한 후 Concat  

- **효과**  
  - 여러 관점에서 단어 간 관계 학습 → 더 정밀한 문맥 이해  
  - 병렬 연산 가능 → 학습 속도 증가  

<img src="/assets/img/bigdatasearch/5/image_10.png" alt="image" width="600px">

---

## p15. 트랜스포머 모델

- **Residual Connection**  
  - 서브층의 입력과 출력을 더하는 것으로 출력과 입력 간의 차이만을 학습  
  - 모델의 입력값이 점진적으로 변화하며 안정적 학습이 되도록 하기 위해 적용  

<img src="/assets/img/bigdatasearch/5/image_11.png" alt="image" width="600px">

- **Residual Connection 결과에 대한 정규화**  
<img src="/assets/img/bigdatasearch/5/image_12.png" alt="image" width="360px"> 
  - 각 Residual Connection 결과 벡터의 평균 μ과 분산 σ²를 구해 정규화를 수행하며, 학습을 안정적으로 수행하도록 함  
<img src="/assets/img/bigdatasearch/5/image_13.png" alt="image" width="360px"> 

---

### 보충 설명  

#### 1. **Residual Connection의 필요성**  
- 신경망이 깊어질수록 기울기 소실(vanishing gradient) 문제가 발생할 수 있다.  
- Residual Connection은 입력을 그대로 더해주는 shortcut 경로를 제공하여, 학습 과정에서 정보 손실을 최소화한다.  
- 이 방식은 네트워크가 "차이(F(x))만 학습"하도록 하여, 학습 안정성과 수렴 속도를 높여준다.  

#### 2. **수식적 해석**  
- 일반적인 학습은 $y = F(x)$로 표현되지만, Residual Connection은  

$$
H(x) = x + F(x)
$$  

  형태로 입력 $x$를 그대로 보존한다.  
- 만약 $F(x) = 0$만 학습하면 $H(x) = x$가 되므로, 네트워크는 곧바로 **항등함수(identity function)**를 구현할 수 있다.  
- 즉, 모델이 원한다면 특정 층을 "아무 일도 하지 않는 층"으로 만들 수 있고, 이는 **깊은 네트워크에서도 성능 저하 없이 안정적인 학습**을 가능하게 한다.  

#### 3. **항등함수(identity function)와 깊은 네트워크**  
- 항등함수는 입력을 그대로 출력하는 함수로, $f(x) = x$ 형태이다.  
- 기존 신경망에서는 입력을 그대로 전달하는 기능을 학습하기 어렵지만, Residual Connection 구조에서는 $F(x)$만 0으로 맞추면 자동으로 구현된다.  
- 이 특성 덕분에 필요 없는 층은 자연스럽게 "우회"할 수 있고, 깊은 네트워크에서도 성능이 떨어지지 않고 유지된다.  

#### 4. **정규화(Layer Normalization)의 역할**  
- Residual Connection 이후 Layer Normalization을 적용하여 평균(μ)과 분산(σ²)을 기준으로 정규화한다.  
- 이를 통해 출력 분포가 일정 범위 안에 유지되어 학습이 불안정해지는 것을 방지한다.  
- 특히 Transformer에서는 각 sublayer(예: Multi-Head Attention, Feed Forward Network)마다 Residual + LayerNorm을 적용해 안정적인 학습을 보장한다.  

---

## p16. Position-Wise FFNN (Feed-Forward Neural Network)  

- 인코더와 디코더에서 공통적으로 갖는 서브층으로 완전 연결 FFNN(Fully-connected FFNN), 동일 가중치를 모든 토큰 위치에 적용  
- Attention 가중합 결과를 비선형 연산을 통해 어텐션 결과를 고차원 공간으로 Projection → 추상화 수준 향상  
- 개별 토큰의 특징을 심화 학습하는 "미세 조정 장치" 역할  
- ReLU 활성화로 하위층에서는 복잡한 단어 내부 패턴 학습(형태소 규칙, 구문특성 등), 상위층에서는 의미론적 관계 등을 학습  

<img src="/assets/img/bigdatasearch/5/image_14.png" alt="image" width="720px"> 

- $X$: 멀티헤드 어텐션 결과 행렬, $(seq\_len, d_{model})$의 크기  
- $W_1$: 가중치 행렬 $(d_{model}, d_{ff})$의 크기  
- $W_2$: 가중치 행렬 $(d_{ff}, d_{model})$의 크기  
- $d_{ff}$: 피드 포워드 신경망의 은닉층의 크기 (논문에서는 2048)  

---

### 보충 설명  

#### 1. **Position-Wise FFNN의 의미**  
- "Position-Wise"라는 말은 각 토큰의 위치마다 **동일한 FFNN이 독립적으로 적용**된다는 것을 의미한다.  
- 즉, 문장의 길이와 상관없이 각 단어 벡터가 같은 구조의 FFNN을 거쳐 변환된다.  
- 이로 인해 모든 단어에 대해 일관된 변환이 보장되며, 문맥을 반영하면서도 위치 독립적인 특성을 유지할 수 있다.  

#### 2. **Projection과 비선형성의 역할**  
- Attention 결과는 단순히 단어 간 관계를 반영한 가중 평균에 불과하다.  
- FFNN은 이 결과를 **비선형 함수(ReLU)** 를 통해 고차원으로 투영(Projection)하여, 더 복잡한 의미적/구문적 패턴까지 학습하게 한다.  

$$  
F_1 = XW_1 + b_1, \quad  
F_2 = \text{ReLU}(F_1), \quad  
F_3 = F_2W_2 + b_2  
$$  

- ReLU는 비선형성을 추가하여 단순한 선형 조합으로는 표현하기 어려운 언어적 특징까지 학습할 수 있도록 한다.  

#### 3. **FFNN의 학습 기능**  
- **하위층**에서는 형태소 규칙, 어휘 패턴 등 **단어 내부의 구조적 특징**을 학습한다.  
- **상위층**에서는 단어 간의 **의미론적 관계(문맥, 추론 관계 등)** 를 학습한다.  
- 따라서 FFNN은 어텐션이 제공한 정보를 더 정제된 특징 공간으로 바꾸는 **미세 조정 장치** 역할을 수행한다.  

#### 4. **차원 확장의 이유 ($d_{ff}$)**  
- Transformer 논문에서는 $d_{ff}=2048$로 설정하여, 입력 차원 $d_{model}=512$보다 훨씬 크게 확장한다.  
- 이렇게 차원을 확장했다가 다시 줄이는 과정은 **더 다양한 특징을 학습**하고 **불필요한 잡음을 줄여 일반화 성능**을 높이는 효과가 있다.  

---

## p17. 트랜스포머 모델  

- **Decoder Block: Masked Self-Attention**  
  - 셀프 어텐션을 통해 어텐션 스코어 행렬 도출 후, 미래에 있는 단어들은 참고하지 못하도록 마스킹  

<img src="/assets/img/bigdatasearch/5/image_15.png" alt="image" width="720px"> 

---

### 보충 설명  

#### 1. **Masked Self-Attention의 필요성**  
- 디코더는 텍스트를 한 단어씩 순차적으로 생성해야 한다.  
- 미래 단어(아직 생성되지 않은 단어)를 미리 참조하면 올바른 학습이 불가능하다.  
- 따라서 마스킹(masking)을 적용하여 현재 시점 이전의 단어까지만 어텐션을 계산한다.  

#### 2. **수식적 해석**  
- 어텐션 스코어 행렬은 Query $Q$와 Key $K$의 내적(dot-product)으로 계산된다.  
- 이때 미래 시점의 단어에 해당하는 스코어는 **$-\infty$ 값으로 마스킹** 처리한다.  
- Softmax 단계에서 $e^{-\infty} = 0$ 이 되므로, 미래 단어는 가중치에 반영되지 않는다.  

$$
\text{Attention}(Q,K,V) = \text{Softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$  

여기서 $M$은 마스크 행렬이며, 미래 단어에 해당하는 위치는 $-\infty$ 값을 가진다.  

#### 3. **훈련 및 추론에서의 역할**  
- **훈련 시**: 모델이 실제 생성 상황과 동일한 조건에서 학습되도록 보장한다.  
- **추론 시**: 이전까지 생성된 단어만을 바탕으로 다음 단어를 예측하여 자연스러운 생성 과정을 유지한다.  

---

## p18. 트랜스포머 모델  

- **Decoder Encode-Decoder MSA**

<img src="/assets/img/bigdatasearch/5/image_16.png" alt="image" width="720px"> 

- Query: 디코더의 첫번째 서브층의 결과 행렬  
- Key: 인코더의 마지막 층에서 온 행렬  
- Value: 인코더의 마지막 층에서 온 행렬  

- **Decoder 입력 시퀀스 데이터의 적절한 위치에 집중할 수 있도록 하기 위함**

---

### 보충 설명  

#### 1. **Encoder-Decoder Attention의 역할**  
- 인코더에서 추출된 문맥 정보(Context Vector)를 디코더가 참조하여 더 정밀한 출력을 생성한다.  
- Query는 디코더에서, Key와 Value는 인코더에서 오기 때문에 입력과 출력 간의 상호작용을 가능하게 한다.  
- 이를 통해 번역이나 요약 같은 작업에서 **입력 문장의 의미를 디코더 출력에 반영**할 수 있다.  

#### 2. **Q, K, V의 의미**  
- **Query (Q)** : 디코더의 현재 상태(부분 출력)가 “어디를 봐야 하는지”를 질의하는 역할.  
- **Key (K)** : 인코더가 입력 문장에서 추출한 위치 및 의미 정보를 제공.  
- **Value (V)** : Key와 연결된 실제 의미 벡터로, Attention 가중치가 곱해져 최종적으로 디코더로 전달됨.  

#### 3. **수식적 해석**  
- Attention은 Query와 Key의 유사도를 계산하여 Softmax로 확률 분포를 만든다.  
- 이후 Value 벡터에 가중합을 취해 최종 출력을 생성한다.  

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$  

#### 4. **장점**  
- 입력과 출력 사이의 **직접적인 연결**을 제공하여 장기 의존성(long-term dependency) 문제를 완화.  
- 특히 기계번역에서 **입력 문장의 특정 단어가 출력 문장의 어떤 단어와 연결되는지**를 학습 가능.  

---

## p19. Linear Layer & Softmax  

- **Linear Layer**  
  - 쌓아올린 디코더들의 출력 결과를 Logits vector로 Projection  
  - 디코더 스택의 최종 출력 벡터를 타겟 어휘 크기(Vocabulary Size)에 맞게 변환  

  - **구조**  
    - 완전 연결계층으로 구성되며,  
    - $Logits = W \cdot h + b$  
      - $W$: 가중치 행렬 (차원: [히든 차원 × 어휘_크기])  
      - $h$: 디코더 출력 벡터 (예: 512차원)  
      - $b$: 편향 벡터  

- **Softmax Layer**  
  - Logits 벡터를 확률 분포로 변환, 최종적으로 분류에 사용할 스코어를 확률로 출력  

<img src="/assets/img/bigdatasearch/5/image_17.png" alt="image" width="720px"> 

---

### 보충 설명  

#### 1. **Linear Layer의 역할**  
- 디코더 스택의 최종 출력 벡터는 그대로 사용할 수 없고, 어휘 집합(vocabulary)에 맞는 크기로 변환해야 한다.  
- Linear Layer는 완전 연결 계층(fully connected layer)으로 구성되어, 디코더 출력 벡터 $h$를 가중치 행렬 $W$와 곱하고 편향 $b$를 더해 **logits**을 계산한다.  
- 수식:  

$$
\text{Logits} = W \cdot h + b
$$  

- 여기서 $W$는 **[히든 차원 × 어휘 크기]**의 행렬, $h$는 디코더 출력 벡터, $b$는 편향 벡터이다.  

#### 2. **Softmax Layer의 역할**  
- Linear Layer에서 얻은 logits는 단순한 실수 값 벡터이므로, 이를 확률 분포로 변환해야 한다.  
- Softmax 함수는 logits를 0과 1 사이의 값으로 변환하고, 전체 합이 1이 되도록 정규화한다.  
- 결과적으로 각 단어가 다음 단어로 선택될 확률을 산출한다.  

#### 3. **Next Word Prediction과의 연결**  
- Transformer 모델은 학습 과정에서 **다음 단어 예측(next word prediction)**을 수행한다.  
- Softmax에서 가장 높은 확률을 갖는 단어가 다음 단어로 예측된다.  
- 이 과정을 반복함으로써 문장 단위의 생성이 가능해진다.  

---

## p20. 트랜스포머 기반 주요 언어모델 유형  

- **Encoder-Only Transformer Model**  
  - 대표모델  
    - BERT (Bidirectional Encoder Representations from Transformers)  
    - ELECTRA  
  - 문장내에서 단어들의 **양방향 문맥을 이해**  
  - 주로 텍스트 분류, 질의응답, 검색 등에 사용  

- **Decoder-Only Transformer Model**  
  - 대표 모델  
    - GPT Family  
    - PaLM Family  
    - LLaMA Family  
  - Auto-Regressive Model로 **언어 생성에 강점**  
  - GPT-3는 1750억 개의 파라미터를 사용  

- **Encoder-Decoder Transformer Model**  
  - 대표모델  
    - T5 (Text-To-Text Transfer Transformer)  
    - BART (Bidirectional and Auto-Regressive Transformers)  
  - 모든 NLP 작업을 텍스트 **변환 문제**로 통합  
  - 다양한 NLP 작업에서 높은 성능  

---

## p21. GPT 모델 시리즈 개요  

<img src="/assets/img/bigdatasearch/5/image_18.png" alt="image" width="720px"> 

- **GPT 모델의 발전과정**  

  - **(2018) GPT-1**  
    - 트랜스포머 기반 초기 모델  
    - 사전 학습과 미세 조정의 시작  

  - **(2019) GPT-2**  
    - 대화 생성 능력의 도약  
    - 제로샷 학습의 가능성 제시  

  - **(2020) GPT-3**  
    - 초대규모 LLM, 퓨샷 학습  

  - **(2022) GPT-3.5 / InstructGPT / ChatGPT**  
    - 인간 피드백 기반 정렬  

  - **(2023) GPT-4**  
    - 멀티모달 및 추론 능력 강화  

  - **(2024) GPT-4o**  
    - 실시간 음성·영상 처리  

  - **(2025) GPT-4.5, 5**  
    - 최신 진화 버전  

- **발전의 방향**  
  1. 모델 규모 증가 → 효율성 최적화 (예: GLaM의 MoE 구조)  
  2. 단순 텍스트 → 멀티모달(이미지·음성) 처리 확장  
  3. 범용 성능 → 도메인 특화(의료, 법률 등) 및 안전성 강조  

---

## p22. GPT 계열 모델 발전 동향  

| 세대   | 출시 시기 | 주요 특징                                | 기술적 진보                                                                 | 활용 범위                                  |
|--------|-----------|------------------------------------------|------------------------------------------------------------------------------|-------------------------------------------|
| GPT1   | 2018      | Transformer 기반 언어 모델 최초 공개     | - 대규모 비지도 사전학습(Pre-training)<br>- 소규모 지도학습(Fine-tuning) 수행 | 자연어이해 중심 실험 수준                  |
| GPT2   | 2019      | 대규모 파라미터(1.5B)로 강력한 생성 능력 확보 | - 단일 모델로 다수의 언어 작업 수행 가능성 제시<br>- Zero-shot, Few-shot 학습 능력 초기 구현 | 텍스트 생성, 요약, 번역, 질의응답          |
| GPT3   | 2020      | 175B 파라미터로 상식 추론·언어 이해 도약 | - 파라미터 수 급증 → 범용 언어 능력 확보<br>- Prompt 기반 학습(지시문 학습) 부상 | 창작, 코딩, 질의응답, 대화형 시스템        |
| GPT3.5 | 2022      | 실용화 단계 진입 (서비스 기반)           | - 인간 피드백 강화학습(RLHF) 적용<br>- 안전성·대화 품질 개선                  | 실시간 대화, 교육, 고객지원                |
| GPT4   | 2023      | 멀티모달 AI로 진화(텍스트+이미지 입력 지원) | - 복합 추론 능력 향상<br>- 안전성·사실성 강화<br>- 도메인 전문성 확장           | 멀티모달 분석, 전문 분야 조언, 복합 추론   |
| GPT-4o | 2024      | ‘Omni’ 모델 – 실시간 멀티모달 대화 지원 | - 음성·이미지·텍스트 처리<br>- 실시간 상호작용 가능<br>- 속도·비용 효율성 개선   | 음성 비서, 실시간 분석, 인터랙티브 에이전트 |
| GPT5   | 2025      | 범용 AGI에 한 걸음 더 접근               | - 장기 추론 및 계획 능력 향상<br>- 에이전트화(지속적 맥락 기억·자율행동)<br>- 다중 작업 협업 능력 강화 | 자율적 업무 처리, 고차원 의사결정 지원     |

---

## p23. GPT 모델 시리즈 개요  

- **공통 기술적 진화 방향**  
  - 모델 규모 증가 → 지능 수준 상승, 효율성 최적화(예: GLaM의 MoE 구조)  
  - Prompt 학습 → Zero/Few-shot 추론 발전  
  - 단순 텍스트 → 멀티모달(이미지·음성) 처리 확장  
    - GPT-4부터 텍스트 외 이미지·음성 등 다양한 입력을 통합적으로 처리하는 단계로 진입  
  - 범용 성능 → 도메인 특화(의료, 법률 등) 및 안전성 강조  
    - 인간 피드백 기반 학습으로 대화 품질, 사용자 친화성, 사실성 개선  
  - 한각 문제 감소를 위한 자기검증(self-checking) 메커니즘 도입  
  - 실시간 대화형 AI → 지능형 에이전트화  
    - GPT-4o 이후 실시간 상호작용 및 다중 작업 수행이 가능해지며, “도구적 AI”에서 “자율적 에이전트”로 진화  

- **한계와 미래 과제**  
  - 한각 문제: 신뢰성 향상을 위한 자기검증 메커니즘 필요  
  - 계산 비용: 에너지 효율적인 학습 알고리즘 개발  
  - 윤리적 문제: 편향 완화와 오용 방지 체계 구축  
