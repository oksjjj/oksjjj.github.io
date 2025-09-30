---
layout: post
title: "[텍스트 마이닝] 6. Representing Texts with Vectors 2"
date: 2025-09-30 10:00:00 +0900
categories:
  - "대학원 수업"
  - "텍스트 마이닝"
tags: []
---

> 출처: 텍스트 마이닝 – 강성구 교수님, 고려대학교 (2025)

# 밀집 고정 표현 (Dense static representation)

- 간단히 하기 위해, 우리는 제품 설명(예: 제목, 특징)을 문서로 지칭한다.  

---

## p12. 문맥에 의한 단어 표현

- 우리가 단어들을 벡터로 표현할 수 있다면, 어떤 텍스트든 그것들을 집계하여 벡터로 표현할 수 있다.  

- **분포 가설 (Distributional hypothesis)**  
  - 비슷한 문맥(context)에서 발생하는 단어들은 비슷한 의미를 가지는 경향이 있다.  
  - 단어의 의미는 그것이 나타나는 문맥에 의해 크게 정의된다.  

<img src="/assets/img/textmining/6/image_1.png" alt="image" width="240px">

- 예시: 우리가 "Ong choy"의 의미를 모른다고 가정하지만, 다음과 같은 문장을 본다고 하자:  
  - Ong choy는 마늘과 함께 볶으면(sautéed with garlic) 맛있다.  
  - Ong choy는 밥 위에(over rice) 얹으면 훌륭하다.  
  - Ong choy는 잎(leaves)과 짠(salty) 소스와 함께한다.  

- 그리고 우리는 다음과 같은 문맥들을 보았다:  
  - … 시금치가 마늘과 함께 볶아져 밥 위에(sautéed with garlic over rice) 올라간다.  
  - … 근대 줄기와 잎(leaves)은 맛있다.  
  - … 콜라드 그린과 다른 짠(salty) 잎채소들.  

---

## p13. 문맥에 의한 단어 표현

- 우리가 단어들을 벡터로 표현할 수 있다면, 어떤 텍스트든 그것들을 집계하여 벡터로 표현할 수 있다.  

- **분포 가설 (Distributional hypothesis)**  
  - 비슷한 문맥(context)에서 발생하는 단어들은 비슷한 의미를 가지는 경향이 있다.  
  - 단어의 의미는 그것이 나타나는 문맥에 의해 크게 정의된다.  

<img src="/assets/img/textmining/6/image_1.png" alt="image" width="240px">

- 예시: 우리가 "Ong choy"의 의미를 모른다고 가정하지만, 다음과 같은 문장을 본다고 하자:  
  - Ong choy는 마늘과 함께 볶으면(sautéed with garlic) 맛있다.  
  - Ong choy는 밥 위에(over rice) 얹으면 훌륭하다.  
  - Ong choy는 잎(leaves)과 짠(salty) 소스와 함께한다.  

- 그리고 우리는 다음과 같은 문맥들을 보았다:  
  - … 시금치가 마늘과 함께 볶아져 밥 위에(sautéed with garlic over rice) 올라간다.  
  - … 근대 줄기와 잎(leaves)은 맛있다.  
  - … 콜라드 그린과 다른 짠(salty) 잎채소들.  

- 이러한 문맥들로부터 우리는 다음을 추론할 수 있다:  
  - Ong choy == water spinach (워터 스피니치, 즉 모닝글로리/공심채).  

---

## p14. 단어 임베딩: 일반적인 아이디어 (Word embeddings: general idea)

- 목표는 **분포 가설(distributional hypothesis)** 에 기반하여 **단어들의 밀집 벡터 표현(dense vector representations of words)** 을 얻는 것이다.  
  - 의미적으로 유사한 단어들(즉, 비슷한 문맥에서 발생하는 단어들)은 유사한 벡터 표현을 가지게 된다.  

- **단어 임베딩(word embeddings)**: 각 행이 하나의 단어에 대한 밀집 벡터에 대응하는 행렬.  
  - 이는 한 공간(어휘, vocabulary)의 원소들을 다른 공간(벡터 공간, vector space)에서 표현하는 사상(mapping)으로 볼 수 있다.  

<img src="/assets/img/textmining/6/image_2.png" alt="image" width="720px">

---

## p15. 단어-문맥 정보 구성 (Constructing word-context information)

- 우리는 단어가 문맥 윈도우(context window) 안에서 다른 단어와 함께 얼마나 자주 나타나는지를 센다.  
  - 윈도우 크기: **<span style="background-color:cyan">타깃 단어(target word)</span>**의 좌우에 있는 **<span style="background-color:yellow">문맥 단어(context words)</span>**의 최대 개수 
  - 예시 (window size = 2)  

<img src="/assets/img/textmining/6/image_3.png" alt="image" width="720px">

---

## p16. 단어-문맥 정보 구성 (Constructing word-context information)

- 공동 발생(co-occurrence) 횟수를 기반으로, **단어-단어 공동 발생 행렬(word-word co-occurrence matrix)** 을 만든다.  
  - 이 행렬의 크기는 $V \times V$이며, 여기서 $V$는 어휘집(vocabulary) 크기이다.  
  - 각 항목은 문맥 윈도우(context window) 안에서 한 단어(행, row)가 다른 단어(열, column)와 함께 얼마나 자주 나타나는지를 센다.  

<img src="/assets/img/textmining/6/image_4.png" alt="image" width="600px">

---

## p17. 단어-문맥 정보 구성 (Constructing word-context information)

- 공동 발생(co-occurrence) 횟수를 기반으로, **단어-단어 공동 발생 행렬(word-word co-occurrence matrix)** 을 만든다.  
  - 이 행렬의 크기는 $V \times V$이며, 여기서 $V$는 어휘집(vocabulary) 크기이다.  
  - 각 항목은 문맥 윈도우(context window) 안에서 한 단어(행, row)가 다른 단어(열, column)와 함께 얼마나 자주 나타나는지를 센다.  

<img src="/assets/img/textmining/6/image_5.png" alt="image" width="600px">

<span style="color:red">이 벡터는 분포 가설(distributional hypothesis)을 반영한다: 의미적으로 유사한 단어들은 유사한 표현을 가진다. 그러나, 이 표현은 희소(sparse)하고 차원이 너무 높다(high-dimensional)!</span>

---

## p18. 밀집 단어 임베딩 생성 (Generating dense word embeddings)

- 이제, 어떻게 이 크고 희소한 행렬을 밀집 단어 임베딩으로 바꿀 수 있을까?  

- 우리는 두 가지 접근법을 논의할 것이다:  

1. **차원 축소 (Dimensionality reduction)**  
   - 공동 발생 행렬(co-occurrence matrix)을 정규화한 후, **truncated SVD**를 적용하여 차원을 축소한다.  
   - 각 행(필요에 따라 특이값(singular values)으로 스케일링됨)은 단어 임베딩으로 사용된다.  

2. **스킵그램 학습 (Skip-gram learning)**  
   - **행렬 분해(matrix factorization)를 수행하는 대신**, 단어-문맥 쌍(word–context pairs)으로부터 직접 임베딩을 학습한다.  
   - 주어진 타깃 단어의 문맥 단어들을 예측하는 모델을 학습한다.  
   - Word2Vec으로 구현된다.  

- 실제로, 이 두 가지 접근법은 수학적으로 동등한 것으로 볼 수 있다.¹  

---

¹ **Neural Word Embedding as Implicit Matrix Factorization, NeurIPS, 2014**

---

## p19. 접근법 1: 차원 축소

- **핵심 아이디어**: 각 희소 벡터(sparse vector)를 저차원 공간으로 사영(projection)하되, **<span style="color:blue">정보를 보존하면서</span>** 수행한다.  

A. **PMI (점별 상호정보량, Pointwise Mutual Information) 재가중치**  
- 앞서 논의했듯이, 단순 빈도(raw frequency)는 의미론을 잘 반영하지 못한다.  
  - 일반적으로 자주 쓰이는 단어들(예: the, it)은 거의 정보를 제공하지 않는다.  
- PMI는 다음을 묻는다:  
  *<span style="color:blue">단어 $w_1$ 과 $w_2$ 는 우연보다 더 자주 함께 등장하는가?</span>*  

$$
PMI(w_1, w_2) = \log \frac{P(w_1, w_2)}{P(w_1) P(w_2)}
$$  

$$
P(w_1, w_2) = \frac{\text{count}(w_1, w_2)}{N}
$$  

$$
P(w_1) = \frac{\text{count}(w_1)}{N}, \quad 
P(w_2) = \frac{\text{count}(w_2)}{N}
$$  

- $N$: 모든 문맥 윈도우(context window)에서의 단어 쌍의 총 개수  
- $\text{count}(w_1, w_2)$: $w_1$ 과 $w_2$ 가 함께 등장한 횟수  
- $\text{count}(w)$: 단어 $w$ 가 등장한 횟수  

- $N$: 모든 문맥 윈도우에서의 단어 쌍의 총 개수  
- $\text{count}(w_1, w_2)$: $w_1$ 과 $w_2$ 가 함께 등장한 횟수  
- $\text{count}(w)$: 단어 $w$ 가 등장한 횟수  

---

## p20. 접근법 1: 차원 축소

- **핵심 아이디어**: 각 희소 벡터(sparse vector)를 저차원 공간으로 사영(projection)하되, **<span style="color:blue">정보를 보존하면서</span>** 수행한다.  

A. **PMI (점별 상호정보량, Pointwise Mutual Information) 재가중치**  

$$
PMI(w_1, w_2) = \log \frac{P(w_1, w_2)}{P(w_1) P(w_2)}
$$  

<img src="/assets/img/textmining/6/image_6.png" alt="image" width="720px">

- <span style="color:red">원시 빈도(raw count)는 4였다</span>  
- <span style="color:red">원시 빈도(raw count)는 7이었다</span>  

- 관례적으로, $PMI < 0$ 인 값은 0으로 표기된다(clipped).  

---

## p21. 접근법 1: 차원 축소

- **핵심 아이디어**: 각 희소 벡터(sparse vector)를 저차원 공간으로 사영(projection)하되, **<span style="color:blue">정보를 보존하면서</span>** 수행한다.  

B. **truncated SVD에 의한 차원 축소 (Dimensionality reduction by truncated SVD)**  
- 축소된 행렬의 각 행(row)은 해당 단어(term)의 단어 임베딩(word embedding)이 된다.  

<img src="/assets/img/textmining/6/image_7.png" alt="image" width="720px">

- $V \times V$ 희소 행렬(sparse matrix)  
- $V \times k$ 밀집 행렬(dense matrix)  

---

## p22. 배경 지식: 차원 축소

- **차원 축소에 대한 직관적인 이해:**  
  - 원래 행렬의 값들이 그렇게 나타나는 이유를 근사적으로 설명해주는 숨겨진(latent) 차원들이 존재한다.  

<img src="/assets/img/textmining/6/image_8.png" alt="image" width="360px">

- **이 차원들의 축(axes of these dimensions)** 은 다음과 같이 선택될 수 있다:  
  - 첫 번째 차원은 데이터가 **<span style="color:blue">가장 큰 분산(greatest variance)</span>** 을 보이는 방향이다.  
  - 두 번째 차원은 첫 번째 차원과 직교하며, **<span style="color:blue">두 번째로 큰 분산(second greatest variance)</span>** 을 포착한다.  
  - 그리고 계속된다 (and so on).  

- **Truncated SVD**는 원래 행렬을 근사할 수 있도록 하며, 데이터를 설명하는 가장 중요한 분산을 포착하는 몇 개의 잠재 차원(latent dimensions)에 사영(projection)한다.  


