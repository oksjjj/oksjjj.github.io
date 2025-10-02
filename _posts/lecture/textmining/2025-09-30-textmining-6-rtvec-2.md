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

<img src="/assets/img/lecture/textmining/6/image_1.png" alt="image" width="240px">

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

<img src="/assets/img/lecture/textmining/6/image_1.png" alt="image" width="240px">

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

<img src="/assets/img/lecture/textmining/6/image_2.png" alt="image" width="720px">

---

## p15. 단어-문맥 정보 구성 (Constructing word-context information)

- 우리는 단어가 문맥 윈도우(context window) 안에서 다른 단어와 함께 얼마나 자주 나타나는지를 센다.  
  - 윈도우 크기: **<span style="background-color:cyan">타깃 단어(target word)</span>**의 좌우에 있는 **<span style="background-color:yellow">문맥 단어(context words)</span>**의 최대 개수 
  - 예시 (window size = 2)  

<img src="/assets/img/lecture/textmining/6/image_3.png" alt="image" width="720px">

---

## p16. 단어-문맥 정보 구성 (Constructing word-context information)

- 공동 발생(co-occurrence) 횟수를 기반으로, **단어-단어 공동 발생 행렬(word-word co-occurrence matrix)** 을 만든다.  
  - 이 행렬의 크기는 $V \times V$이며, 여기서 $V$는 어휘집(vocabulary) 크기이다.  
  - 각 항목은 문맥 윈도우(context window) 안에서 한 단어(행, row)가 다른 단어(열, column)와 함께 얼마나 자주 나타나는지를 센다.  

<img src="/assets/img/lecture/textmining/6/image_4.png" alt="image" width="600px">

---

## p17. 단어-문맥 정보 구성 (Constructing word-context information)

- 공동 발생(co-occurrence) 횟수를 기반으로, **단어-단어 공동 발생 행렬(word-word co-occurrence matrix)** 을 만든다.  
  - 이 행렬의 크기는 $V \times V$이며, 여기서 $V$는 어휘집(vocabulary) 크기이다.  
  - 각 항목은 문맥 윈도우(context window) 안에서 한 단어(행, row)가 다른 단어(열, column)와 함께 얼마나 자주 나타나는지를 센다.  

<img src="/assets/img/lecture/textmining/6/image_5.png" alt="image" width="600px">

<span style="color:red">이 벡터는 분포 가설(distributional hypothesis)을 반영한다: 의미적으로 유사한 단어들은 유사한 표현을 가진다. 그러나, 이 표현은 희소(sparse)하고 차원이 너무 높다(high-dimensional)!</span>

---

## p18. 밀집 단어 임베딩 생성 (Generating dense word embeddings)

- 이제, 어떻게 이 크고 희소한 행렬을 밀집 단어 임베딩으로 바꿀 수 있을까?  

- 우리는 두 가지 접근법을 논의할 것이다:  

1. **<span style="color:blue">차원 축소 (Dimensionality reduction)</span>**  
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

<img src="/assets/img/lecture/textmining/6/image_6.png" alt="image" width="720px">

- <span style="color:red">원시 빈도(raw count)는 4였다</span>  
- <span style="color:red">원시 빈도(raw count)는 7이었다</span>  

- 관례적으로, $PMI < 0$ 인 값은 0으로 표기된다(clipped).  

---

## p21. 접근법 1: 차원 축소

- **핵심 아이디어**: 각 희소 벡터(sparse vector)를 저차원 공간으로 사영(projection)하되, **<span style="color:blue">정보를 보존하면서</span>** 수행한다.  

B. **truncated SVD에 의한 차원 축소 (Dimensionality reduction by truncated SVD)**  
- 축소된 행렬의 각 행(row)은 해당 단어(term)의 단어 임베딩(word embedding)이 된다.  

<img src="/assets/img/lecture/textmining/6/image_7.png" alt="image" width="720px">

- $V \times V$ 희소 행렬(sparse matrix)  
- $V \times k$ 밀집 행렬(dense matrix)  

---

## p22. 배경 지식: 차원 축소

- **차원 축소에 대한 직관적인 이해:**  
  - 원래 행렬의 값들이 그렇게 나타나는 이유를 근사적으로 설명해주는 숨겨진(latent) 차원들이 존재한다.  

<img src="/assets/img/lecture/textmining/6/image_8.png" alt="image" width="360px">

- **이 차원들의 축(axes of these dimensions)** 은 다음과 같이 선택될 수 있다:  
  - 첫 번째 차원은 데이터가 **<span style="color:blue">가장 큰 분산(greatest variance)</span>** 을 보이는 방향이다.  
  - 두 번째 차원은 첫 번째 차원과 직교하며, **<span style="color:blue">두 번째로 큰 분산(second greatest variance)</span>** 을 포착한다.  
  - 그리고 계속된다 (and so on).  

- **Truncated SVD**는 원래 행렬을 근사할 수 있도록 하며, 데이터를 설명하는 가장 중요한 분산을 포착하는 몇 개의 잠재 차원(latent dimensions)에 사영(projection)한다.  

---

## p23. 배경 지식: Truncated SVD

- **<span style="color:blue">특이값 분해(SVD, Singular Value Decomposition)</span>**:  
  데이터가 $n \times d$ 행렬 $A$로 표현될 때, 이는 세 개의 ‘단순한(simple)’ 행렬의 곱으로 나타낼 수 있다.  

$$
A = U S V^T
$$  

<img src="/assets/img/lecture/textmining/6/image_9.png" alt="image" width="600px">

- $n$: 데이터 인스턴스(data instances)의 개수  
- $d$: 원래 특성(original features)의 개수  

*SVD는 선형대수학 강의에서 다루어지는 내용이다. 혹시 잊었다면, 다음 참고 문헌을 보라 (for review):  
<a href="https://web.stanford.edu/class/cs168/l/l9.pdf" target="_blank">Stanford CS168 강의노트</a>*  
[**SVD 강의 한국어 번역**](/posts/math-svd/){:target="_blank"}

---

## p24. 배경 지식: Truncated SVD

- **<span style="color:blue">Truncated SVD</span>**:  
  행렬 $A$는 세 개의 행렬 곱으로 <span style="color:blue">가깝게 근사될 수 있다 (closely approximated)</span>.  
  이때 세 행렬은 더 작은 공통 차원 $k$를 공유한다.  

$$
A \approx U_k S_k V_k^T
$$  

<img src="/assets/img/lecture/textmining/6/image_10.png" alt="image" width="600px">

- $n$: 데이터 인스턴스(data instances)의 개수  
- $d$: 원래 특성(original features)의 개수  
- $k$: 유지되는 잠재(latent) 차원의 개수 ($k < d$)  

---

## p25. 배경 지식: Truncated SVD

- **Truncated SVD**: 행렬 A는 더 작은 공통 차원(k)을 공유하는 세 개의 행렬 곱으로 **가까이 근사(closely approximated)** 될 수 있다.  

- **예시 (Example):**

<img src="/assets/img/lecture/textmining/6/image_11.png" alt="image" width="720px">

---

## p26. 밀집 단어 임베딩 생성 (Generating dense word embeddings)

- *이제, 어떻게 이 크고 희소한 행렬을 밀집 단어 임베딩으로 변환할 수 있을까?*  

- 우리는 두 가지 접근 방식을 논의할 것이다:  

1. **차원 축소 (Dimensionality reduction)**  
   - 동시발생 행렬(co-occurrence matrix)을 정규화한 후, **Truncated SVD**를 적용하여 차원을 축소한다.  
   - 각 행은 (선택적으로 특이값으로 스케일된) 단어 임베딩으로 사용된다.  

2. **<span style="color:blue">Skip-gram 학습</span>**  
   - 행렬을 인수분해하는 대신, **단어-문맥 쌍(word–context pairs)**으로부터 직접 임베딩을 학습한다.  
   - 주어진 목표 단어(target word)의 문맥 단어(context words)를 예측하는 모델을 훈련한다.  
   - Word2Vec으로 구현된다.  

- 실제로, 이 두 접근 방식은 수학적으로 동등한 것으로 볼 수 있다.  

<sub>Neural Word Embedding as Implicit Matrix Factorization, NeurIPS, 2014</sub>

---

## p27. 요약: 모델, 손실 함수, 최적화

<img src="/assets/img/lecture/textmining/6/image_12.png" alt="image" width="720px">

- 어떤 예측기(predictors)가 가능한가?  
  - **가설 집합 (Hypothesis class)**  

    $$
    \mathcal{F} = \{ f_{\mathbf{w}}(x) = \mathbf{w} \cdot \varphi(x) \}
    $$  

- 예측기가 얼마나 좋은가?  
  - **손실 함수 (Loss function)**  

    $$
    Loss(x, y, \mathbf{w}) = (f_{\mathbf{w}}(x) - y)^2
    $$  

- 예측기를 어떻게 최적화할 것인가?  
  - **경사 하강법 (Gradient descent)**  

    $$
    \mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} \, TrainLoss(\mathbf{w})
    $$  

---

## p28. 접근 방식 2: Skip-gram 학습

- **핵심 아이디어 (Key idea)**:  
  **<span style="background-color:yellow">맥락 단어들(the context words)</span>**을  
  **<span style="background-color:cyan">중심 단어(the center word)</span>**를 사용하여 예측한다.  
  (의미적으로 유사한 중심 단어들은 유사한 맥락 단어들을 예측한다.)

> *“Skip-gram”: 일부 맥락 단어들을 건너뛰고(context words), 나머지를 예측한다!*

<img src="/assets/img/lecture/textmining/6/image_13.png" alt="image" width="720px">

<span style="color:red">맥락 창(context window) 밖의 단어들을 건너뛰기, 예: new, pop</span>

---

## p29. 접근 방식 2: Skip-gram 학습

- **핵심 아이디어 (Key idea)**:  
  **<span style="background-color:yellow">맥락 단어들(the context words)</span>**을  
  **<span style="background-color:cyan">중심 단어(the center word)</span>**를 사용하여 예측한다.  
  (의미적으로 유사한 중심 단어들은 유사한 맥락 단어들을 예측한다.)

- **목표 (Objective)**:  
  **<span style="background-color:yellow">맥락 단어 c(the context word c)</span>**를  
  **<span style="background-color:cyan">중심 단어 w(the center word w)</span>**를 사용하여 예측할 확률을 극대화한다.

$$
\max_{\theta} \prod_{(w,c) \in D} p_{\theta}(c \mid w)
$$

- *D*: 전체 공기출현 쌍 집합 (total set of co-occurrence pairs)  
- *θ*: 최적화될 단어 임베딩들 (word embeddings to be optimized, model parameters)

<img src="/assets/img/lecture/textmining/6/image_14.png" alt="image" width="360px">

- 확률을 어떻게 표현할까? (How to express the probability?)

---

## p30. 접근 방식 2: Skip-gram 학습

- **핵심 아이디어 (Key idea)**:  
  **<span style="background-color:yellow">맥락 단어들(the context words)</span>**을  
  **<span style="background-color:cyan">중심 단어(the center word)</span>**를 사용하여 예측한다.  
  (의미적으로 유사한 중심 단어들은 유사한 맥락 단어들을 예측한다.)

- **목표 (Objective)**:  
  **<span style="background-color:yellow">맥락 단어 c(the context word c)</span>**를  
  **<span style="background-color:cyan">중심 단어 w(the center word w)</span>**를 사용하여 예측할 확률을 극대화한다.

$$
\max_{\theta} \prod_{(w,c) \in D} p_{\theta}(c \mid w)
$$

- *D*: 전체 공기출현 쌍 집합 (total set of co-occurrence pairs)  
- *θ*: 최적화될 단어 임베딩들 (word embeddings to be optimized)  
  { $v_w, v_c$ }

- 각 단어는 두 개의 밀집 벡터(dense vectors)를 할당받는다:  
  - 중심 단어 역할(center-word role)을 위한 것  
  - 맥락 단어 역할(context-word role)을 위한 것  

- 로그 확률(log-probability)은 **벡터 내적(vector inner product)**에 비례한다고 가정한다:

$$
\log p_{\theta}(c \mid w) \propto v_c \cdot v_w
$$

<img src="/assets/img/lecture/textmining/6/image_15.png" alt="image" width="360px">

---

## p31. 접근 방식 2: Skip-gram 학습

- **목표 (Objective)**:  
  **<span style="background-color:yellow">맥락 단어 c(the context word c)</span>**를  
  **<span style="background-color:cyan">중심 단어 w(the center word w)</span>**를 사용하여 예측할 확률을 극대화한다.

$$
\max_{\theta} \prod_{(w,c) \in D} p_{\theta}(c \mid w)
$$

- *D*: 전체 공기출현 쌍 집합 (total set of co-occurrence pairs)  
- *θ*: 최적화될 단어 임베딩들 (word embeddings to be optimized)  
  { $v_w, v_c$ }

---

- 로그 확률(log-probability)은 **벡터 내적(vector inner product)**에 비례한다고 가정한다:

  - 각 $(w, c)$ 쌍에 대해:

  $$
  \log p_{\theta}(c \mid w) \propto v_c \cdot v_w
  $$

  - 다른 모든 맥락 단어들 $(c')$에 대해:

  $$
  \log p_{\theta}(c' \mid w) \propto v_{c'} \cdot v_w
  $$

<img src="/assets/img/lecture/textmining/6/image_16.png" alt="image" width="360px">

---

## p32. 접근 방식 2: Skip-gram 학습

- **목표 (Objective)**:  
  **<span style="background-color:yellow">맥락 단어 c(the context word c)</span>**를  
  **<span style="background-color:cyan">중심 단어 w(the center word w)</span>**를 사용하여 예측할 확률을 극대화한다.

$$
\max_{\theta} \prod_{(w,c) \in D} p_{\theta}(c \mid w)
$$

- *D*: 전체 공기출현 쌍 집합 (total set of co-occurrence pairs)  
- *θ*: 최적화될 단어 임베딩들 (word embeddings to be optimized) { $v_w, v_c$ }

---

- 최종 확률분포(final probability distribution)는 **소프트맥스 함수(softmax function)**로 얻어진다:

$$
p_{\theta}(c \mid w) = \frac{\exp(v_c \cdot v_w)}{\sum_{c' \in |V|} \exp(v_{c'} \cdot v_w)}
$$

<span style="color:red">어휘 전체(vocabulary)에 대해 정규화(normalize)하여 확률 분포를 만든다</span>

---

<img src="/assets/img/lecture/textmining/6/softmax.png" alt="image" width="360px">

<img src="/assets/img/lecture/textmining/6/image_17.png" alt="image" width="360px">

---

## p33. 접근 방식 2: Skip-gram 학습

- 우리는 단어 임베딩(word embeddings)을 최적화 문제(negative log)로 정식화하였다:

$$
\min_{\theta} \mathcal{L}(\theta) 
= - \sum_{(w,c) \in D} \log p_{\theta}(c \mid w) 
= - \sum_{(w,c) \in D} 
\left( v_c \cdot v_w - \log \sum_{c' \in |V|} \exp(v_{c'} \cdot v_w) \right)
$$

- *D*: 전체 공기출현 쌍 집합 (total set of co-occurrence pairs)  
- *θ*: 최적화될 단어 임베딩들 (word embeddings to be optimized) { $v_w, v_c$ }

---

$$
p_{\theta}(c \mid w) 
= \frac{\exp(v_c \cdot v_w)}{\sum_{c' \in |V|} \exp(v_{c'} \cdot v_w)}
$$

<img src="/assets/img/lecture/textmining/6/image_18.png" alt="image" width="360px">

---

## p34. 접근 방식 2: Skip-gram 학습

- 우리는 단어 임베딩(word embeddings)을 최적화 문제(negative log)로 정식화하였다:

$$
\min_{\theta} \mathcal{L}(\theta) 
= - \sum_{(w,c) \in D} \log p_{\theta}(c \mid w) 
= - \sum_{(w,c) \in D} 
\left( v_c \cdot v_w - \log \sum_{c' \in |V|} \exp(v_{c'} \cdot v_w) \right)
$$

---

- **어떻게 최적화할까? (How to optimize?)**: 경사하강법(gradient descent)  
  - 그래디언트 $\nabla_{\theta} \mathcal{L}(\theta)$는 훈련 손실을 가장 크게 증가시키는 방향이다.

---

💻 **알고리즘 (Algorithm)**  
- 무작위로 $\theta$ 초기화  
- 수렴할 때까지 반복:  

$$
\theta \leftarrow \theta - \eta \nabla_{\theta} \mathcal{L}(\theta)
$$

- $\theta$: 학습(또는 모델) 매개변수 (learning/model parameters)  
- $\eta$: 단계 크기(step size, 학습률 learning rate) → 하이퍼파라미터  

<img src="/assets/img/lecture/textmining/6/image_19.png" alt="image" width="480px">

---

## p35. 배경 지식: 경사하강법 (Gradient descent)

- 우리는 목적 함수(손실 함수) $\mathcal{L}(\theta)$를 가진다.  

- **경사하강법(Gradient descent)** 은 $\mathcal{L}(\theta)$를 최소화하기 위해 사용되는 반복적 최적화 알고리즘이다.  

---

- **어떻게 작동하는가? (How it works?)**  
  현재 값 $\theta$에 대해 $\mathcal{L}(\theta)$의 그래디언트를 계산한 다음, 음의 그래디언트 방향으로 작은 걸음을 이동한다.  
  이를 반복한다.  

---

💻 **알고리즘 (Algorithm)**  
- 무작위로 $\theta$ 초기화  
- 수렴할 때까지 반복:  

$$
\theta \leftarrow \theta - \eta \nabla_{\theta} \mathcal{L}(\theta)
$$

---

- $\theta$: 학습 매개변수 (learning parameters)  
- $\eta$: 단계 크기(step size, 학습률 learning rate) → 하이퍼파라미터  

---

<img src="/assets/img/lecture/textmining/6/image_20.png" alt="image" width="480px">

---

📎 **참고:** <a href="https://cs231n.stanford.edu/handouts/derivatives.pdf" target="_blank">Notes on derivatives (Stanford CS231n)</a>

---

## p36. 접근 방식 2: Skip-gram 학습

- 우리의 문제로 돌아와서, 손실(loss)을 최소화하기 위해 학습 매개변수 $\theta$를 업데이트한다.  

$$
\min_{\theta} \; \mathcal{L}(\theta) 
= - \sum_{(w,c) \in D} \left( \mathbf{v}_c \cdot \mathbf{v}_w - \log \sum_{c' \in |\mathcal{V}|} \exp(\mathbf{v}_{c'} \cdot \mathbf{v}_w) \right)
$$

---

- **예시 (E.g.)**, 쌍 ( <span style="background-color:cyan">Taylor</span>, <span style="background-color:yellow">release</span> ):  

  - <span style="color:blue">양성(실제) 쌍(positive (real) pairs)</span>의 내적(inner product)을 **최대화**한다.  
  - <span style="color:red">음성 쌍(negative pairs)</span>의 내적(inner product)을 **최소화**한다.  

---

<img src="/assets/img/lecture/textmining/6/image_21.png" alt="image" width="720px">

---

## p37. 접근 방식 2: Skip-gram 학습 (음성 샘플링, Negative Sampling)

- 이전 해법은 동작하지만, **비효율적**이다.  

$$
\min_{\theta} \; \mathcal{L}(\theta) 
= - \sum_{(w,c) \in D} \left( \mathbf{v}_c \cdot \mathbf{v}_w 
- \log \sum_{c' \in |\mathcal{V}|} \exp(\mathbf{v}_{c'} \cdot \mathbf{v}_w) \right)
$$  

<span style="color:red">어휘집 전체(vocabulary)에 대해 합(sum)을 수행해야 하므로 비용이 많이 든다 (expensive).</span>

---

<img src="/assets/img/lecture/textmining/6/image_22.png" alt="image" width="360px">

---

## p38. 접근 방식 2: Skip-gram 학습 (부정 샘플링, negative sampling)

- 이전 해결책은 동작하지만, 비효율적이다.  

$$
\min_{\theta} \ \mathcal{L}(\theta) \;=\; - \sum_{(w,c) \in D} 
\Bigg( \, \mathbf{v}_c \cdot \mathbf{v}_w \;-\; \log \sum_{c' \in |\mathcal{V}|} \exp(\mathbf{v}_{c'} \cdot \mathbf{v}_w) \,\Bigg)
$$  

<span style="color:red">어휘집 전체(vocabulary)에 대해 합산해야 함 → 비용이 큼 (expensive!)</span>  

---

**Negative sampling**
- 우리는 소수의 “부정 단어들(negative words)”만 샘플링하여 업데이트에 사용한다.  

<img src="/assets/img/lecture/textmining/6/image_23.png" alt="image" width="720px">

---

## p39. 접근 방식 2: Skip-gram 학습 (부정 샘플링, negative sampling)

- 이전 해결책은 동작하지만, 비효율적이다.  

$$
\min_{\theta} \ \mathcal{L}(\theta) \;=\; - \sum_{(w,c) \in D} 
\Bigg( \, \mathbf{v}_c \cdot \mathbf{v}_w \;-\; \log \sum_{c' \in |\mathcal{V}|} \exp(\mathbf{v}_{c'} \cdot \mathbf{v}_w) \,\Bigg)
$$  

<span style="color:red">어휘집 전체(vocabulary)에 대해 합산해야 함 → 비용이 큼 (expensive!)</span>  

---

- 따라서 우리는 **근사 목적 함수(approximate objective)**를 사용하여 더 쉽게 최적화한다.  
  - 목적 함수를 **이진 분류(binary classification)** 과제로 다시 정의한다:  
    $(w, c)$가 **진짜 쌍(true pair)**인지 예측한다.  
  - $(w, c)$가 진짜 쌍일 확률은 **시그모이드 함수(sigmoid function)**로 계산된다.  

$$
p_\theta(\text{True} \mid c, w) \;=\; \sigma(\mathbf{v}_c \cdot \mathbf{v}_w) 
\;=\; \frac{1}{1 + \exp(- \mathbf{v}_c \cdot \mathbf{v}_w)}
$$  

---

<img src="/assets/img/lecture/textmining/6/image_24.png" alt="image" width="480px">

---

$$
p_\theta(\text{False} \mid c, w) 
= 1 - p_\theta(\text{True} \mid c, w) 
= 1 - \sigma(\mathbf{v}_c \cdot \mathbf{v}_w) 
= \sigma(- \mathbf{v}_c \cdot \mathbf{v}_w)
$$  

---

## p40. 접근 방식 2: Skip-gram 학습 (부정 샘플링, negative sampling)

- 따라서 우리는 **근사 목적 함수(approximate objective)**를 사용하여 더 쉽게 최적화한다.  
  - **양의 쌍(positive pairs, 실제로 함께 등장하는 단어들)**의 확률을 최대화한다.  
  - **음의 쌍(negative pairs, 무작위로 샘플링된 잡음)**의 확률을 최소화한다.  

---

**Softmax vs. Sigmoid**  

- *Softmax*: 가능한 모든 쌍들 중에서, **어떤 쌍이 가장 가능성이 높은가?**  
- *Sigmoid*: 특정 쌍에 대해서, **이 쌍이 진짜인가 가짜인가?**  

---

**목적 함수 변화**  

$$
\min_{\theta} \ \mathcal{L}(\theta) \;=\; - \sum_{(w,c) \in D} 
\Bigg( \mathbf{v}_c \cdot \mathbf{v}_w \;-\; \log \sum_{c' \in |\mathcal{V}|} \exp(\mathbf{v}_{c'} \cdot \mathbf{v}_w) \Bigg)
$$  

⬇️  

$$
\min_{\theta} \ \mathcal{L}(\theta) \;=\; - \sum_{(w,c) \in D} 
\Bigg( \log \sigma(\mathbf{v}_c \cdot \mathbf{v}_w) 
\;+\; \sum_{c' \in \mathcal{N}} \log \sigma(- \mathbf{v}_{c'} \cdot \mathbf{v}_w) \Bigg)
$$  

- <span style="color:blue">positive pairs</span>: $\log \sigma(\mathbf{v}_c \cdot \mathbf{v}_w)$  
- <span style="color:red">negative pairs</span>: $\sum \log \sigma(- \mathbf{v}_{c'} \cdot \mathbf{v}_w)$  

---

- $\mathcal{N}$: 무작위로 샘플링된 음의 집합 (크기는 $ \mid \mathcal{V} \mid $보다 작음, 보통 5–10개)  

---

## p41. Word2Vec 시각화 (Word2Vec visualization)

- **데모(Demo)**:  
  <a href="https://projector.tensorflow.org/" target="_blank">https://projector.tensorflow.org/</a>  

<img src="/assets/img/lecture/textmining/6/image_25.png" alt="image" width="720px">

---

## p42. 요약: 밀집 고정 표현 (Dense static representation)

- **분포 가설 (Distributional hypothesis)**  
  - 유사한 문맥(context)에서 나타나는 단어들은 유사한 의미를 가지는 경향이 있다.  

- 단어–문맥 쌍(word–context pairs) (혹은 행렬 형태) 이 주어졌을 때, 두 가지 주요 접근법이 있다:  

1. **차원 축소 (Dimensionality reduction)**  
   - 공출현 행렬(co-occurrence matrix)로부터 시작한다.  
   - 정규화(normalize)하고 **절단된 SVD (truncated SVD)**를 적용하여 차원을 축소한다.  
   - 각 행(필요시 특이값으로 스케일링된)은 단어 임베딩(word embedding)으로 사용된다.  

2. **스킵그램 학습 (Skip-gram learning)**  
   - 단어–문맥 쌍(word–context pairs)으로부터 직접 임베딩을 학습한다.  
   - 주어진 목표 단어(target word)의 문맥 단어(context words)를 예측한다.  
   - 학습을 효율적으로 만들기 위해 네거티브 샘플링(negative sampling)을 사용한다.  
   - Word2Vec으로 구현된다.  

---

## p43. 밀집 표현들의 유사성 (Similarity of dense representations)

- 단어 임베딩(word embeddings)을 사용하면, 어떤 텍스트든 이를 집계(예: 평균)하여 벡터로 표현할 수 있다.  

- 쿼리(query)와 문서(document)가 주어졌을 때, 우리는 밀집 표현(dense representations)에 기반하여 그들의 유사성을 계산한다:  
  - **<span style="color:blue">쿼리 q</span>**: “Taylor release new album”  
  - **<span style="color:purple">문서 d</span>**: “American singer Taylor …”  

<img src="/assets/img/lecture/textmining/6/image_26.png" alt="image" width="720px">

---

## p44. 밀집 표현들의 유사성 (Similarity of dense representations)

- 단어 임베딩(word embeddings)을 사용하면, 어떤 텍스트든 이를 집계(예: 평균)하여 벡터로 표현할 수 있다.  

- 쿼리(query)와 문서(document)가 주어졌을 때, 우리는 밀집 표현(dense representations)에 기반하여 그들의 유사성을 계산한다.  

- **내적(inner product)** 과 **코사인 유사도(cosine similarity)** 가 여기에서 널리 사용된다.  

---

- **희소 벡터(Sparse vectors):**  
  - 고차원이며, 대부분이 0  
  - <span style="color:red">크기는 문서 길이에 의존</span>  
  - 정규화 필요  
  - ➤ **코사인 유사도가 선호됨**  

- **밀집 벡터(Dense vectors):**  
  - 저차원, 연속적인 값  
  - <span style="color:blue">노름(norms)은 훨씬 더 안정적</span>  
  - 단어 임베딩 풀링(pooling)은 → <span style="color:blue">길이의 영향을 덜 받음</span>  
  - ➤ **내적(inner product)** 은 유사도 측정으로 사용될 수 있음  
  - 노름 불변성을 선호하는 경우 코사인 유사도도 가능  

---

---

## p38. 다음: 밀집 "맥락" 표현 (Dense "contextual" representations)

- **희소 표현(sparse representation)에서 밀집 표현(dense representation)으로**  

  - 희소 벡터(sparse vectors): 매우 길다 (길이 = $ \mid V \mid $, 종종 10k 이상), 대부분의 항목 값 = 0  
  - 밀집 벡터(dense vectors): 상대적으로 짧다 (50–1000 차원), 대부분의 항목 값 ≠ 0  

---

- **정적 임베딩(Static embeddings)**  
  - 각 단어는 **하나의 고정된 밀집 벡터(single fixed dense vector)** 로 할당된다.  
  - 주변 문맥(context)을 반영하지 않는다.  
    - 예: “bank” → 항상 같은 벡터  
  - 예시: Word2vec, GloVe  

  <img src="/assets/img/lecture/textmining/5/image_26.png" alt="image" width="300px">  

---

- **문맥 임베딩(Contextual embeddings)**  
  - 각 단어의 벡터는 **주변 문맥(surrounding context)** 에 따라 달라진다.  
  - 단어의 의미가 문맥에 따라 변한다.  
    - 예: “bank of the river” vs. “bank account”  
  - 예시: BERT, LLM 기반 임베딩  

  <img src="/assets/img/lecture/textmining/5/image_27.png" alt="image" width="500px">  

---

## p46. 추천 읽을거리 (Recommended readings)

- **Speech and Language Processing**: *Chapter 5: Embeddings*
