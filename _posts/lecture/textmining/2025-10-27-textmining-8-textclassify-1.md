---
layout: post
title: "[텍스트 마이닝] 8. Text Classification 1"
date: 2025-10-27 11:00:00 +0900
categories:
  - "대학원 수업"
  - "텍스트 마이닝"
tags: []
---

> 출처: 텍스트 마이닝 – 강성구 교수님, 고려대학교 (2025)

## p9. 텍스트 분류 (Text classification)

- 왜 중요한가?  
  - 데이터 분석의 핵심 과제: **미리 정의된 레이블을 할당(assign predefined labels)** (예: 브랜드, 감정, 주제 등)  
  - 검색 엔진, 추천 시스템, 스팸 탐지 등에서 폭넓게 사용된다.  
  - QA, 대화 시스템, 개인화(personalization) 등 다양한 고급 응용의 기초가 된다.  

<img src="/assets/img/lecture/textmining/8/image_1.png" alt="image" width="800px">

- 우리의 학습 경로:  
  - 텍스트를 벡터로 표현하였다.  
  - 다음으로, 이러한 표현을 이용하여 **분류(classification)** 를 수행할 것이다.  
  - 주로 **사전학습(pretrain) - 미세조정(fine-tune) 패러다임** 을 따라, 일반적인 언어 지식을 특정 분류 과제에 맞게 조정할 것이다.  

---

# p10. 사전학습 및 미세조정

---

## p11. 사전학습 + 미세조정 (Pretraining + Fine-tuning)

- 사전학습(Pretraining)  
  - 동기(Motivation): 웹에는 언어적 패턴과 세계 지식이 풍부하게 담긴 방대한 양의 텍스트 데이터가 존재한다.  
  - 목표: 사람의 레이블링 없이 **일반적인 목적의 표현(general-purpose representations)** 을 학습하는 것.  
  - 예시:  
    - Word2Vec (인접 단어 예측)  
    - BERT (마스크된 언어 모델링, 다음 문장 예측)  

<img src="/assets/img/lecture/textmining/8/image_2.png" alt="image" width="800px">

---

## p12. 사전학습 + 미세조정

- 미세조정(Fine-tuning)  
  - 동기(Motivation): 사전학습된 모델은 일반적인 지식을 포착하지만, 분류와 같은 작업에는 과제(task) 특화 지식이 필요하다.  
  - 목표: **사전학습된 모델의 파라미터를 조정(adjust the pretrained model parameters)** 하여 **특정 목표(downstream) 과제** 에서 좋은 성능을 내도록 하는 것.  
    - 처음부터 학습하는 것보다 훨씬 적은 레이블된 데이터(labeled data)만 필요하다.  
    - 사전학습 과정에서 학습된 지식을 활용한다.  
  - 예시:  
    - BERT를 이용한 감정 분류(sentiment classification)  

<img src="/assets/img/lecture/textmining/8/image_3.png" alt="image" width="800px">

---

## p13. 사전학습 + 미세조정

- 왜 사전학습 + 미세조정이 “최적화 관점(optimization perspective)”에서 도움이 되는가?  

- 사전학습(Pretraining)  
  - 손실함수 $L_{\text{pretrain}}$ 을 최소화하여 파라미터 $\hat{\theta}$ 를 학습한다.  
  - 좋은 초기값(good initialization)을 제공한다.  

- 미세조정(Fine-tuning)  
  - $\hat{\theta}$ 에서 시작하여 손실함수 $L_{\text{fine-tun}}$ 을 최소화한다.  
  - 사전학습된 모델을 목표 과제(target task)에 맞게 적응시킨다.  

- 확률적 경사 하강법(SGD)은 초기값(initialization)에 크게 영향을 받는다.  
- 사전학습으로부터 좋은 시작점을 얻으면,  
  모델은 효율적으로 수렴(converge efficiently)하며  
  더 적은 양의 데이터로도 잘 일반화(generalize well)되는 경향이 있다.  

<img src="/assets/img/lecture/textmining/8/image_4.png" alt="image" width="600px">

---

## p14. 사전학습 + 미세조정 예시

- Downstream task: 감정 분류  
  - 주어진 문장이 긍정(positive), 중립(neutral), 부정(negative)인지 예측한다.  

- 작동 방식:  
  - 사전학습된 BERT를 이용해 각 문장을 벡터로 표현한다.  
    - 일반적인 선택: CLS 표현(Representation) 또는 모든 문맥 임베딩의 평균.  
  - 작은 분류 헤드(classification head, 작은 신경망)를 추가한다.  
  - 이후 모델을 레이블이 있는 데이터(labeled data)로 미세조정(fine-tuning)한다.  

- 학습 선택:  
  A. 전체 미세조정(Full fine-tuning): 모든 파라미터(all parameters)를 업데이트한다.  
  B. 부분 미세조정(Partial fine-tuning): 일부 파라미터(subset of parameters)만 업데이트한다.  
     - 인코더(encoder)를 고정(freeze)하고 **헤드만 업데이트(only the head)** 한다.  
     - 인코더 대부분을 고정하고 **상위 층(top layers)** 과 **헤드(head)** 를 함께 업데이트한다.  

<img src="/assets/img/lecture/textmining/8/image_5.png" alt="image" width="800px">

---

# p15. 분류 작업(Classification task)

---

## p16. 분류: 이진 분류 vs 다중 클래스 분류 

- **분류(Classification)**: 미리 정의된 범주 집합 $C$에서 **이산적인 레이블**을 예측하는 것  

$$
x \;\longrightarrow\; f \;\longrightarrow\; y \in C
$$

- 이진 분류 (Binary classification, $\mid C \mid = 2$)  
  - 예: 스팸 탐지 (스팸 / 정상 메일)  
<img src="/assets/img/lecture/textmining/2/image_3.png" alt="image" width="480px">

- 다중 클래스 분류 (Multiclass classification, $\mid C \mid > 2$)  
  - 예: 이미지 분류 (고양이, 개, 말)  
<img src="/assets/img/lecture/textmining/2/image_4.png" alt="image" width="480px">

---

## p18. 분류: 분류기(classifier)

- 이제 우리는 (1) 사전학습된 모델을 사용하여 텍스트를 **벡터 표현(vector representation)** 으로 인코딩하고,  
  (2) 각 클래스에 대한 **확률(probability)** 을 계산할 수 있다.  

<img src="/assets/img/lecture/textmining/8/image_6.png" alt="image" width="800px">

---

## p19. 분류: 손실 함수 (이진)

- 우리는 다음을 가진다  

  $$
  \mathcal{D} = \{(x_n, y_n) \mid n = 1, \ldots, N\}, \quad \text{where } y_i \in \{0, 1\}
  $$

- 손실 유도(Loss derivation):  
  - 각 레이블은 베르누이 확률변수(Bernoulli random variable)로 모델링될 수 있다.  
  >✓ 베르누이 분포(Bernoulli distribution):  
  >  - 확률변수 $y = 1$ 은 확률 $p$ 로, $y = 0$ 은 확률 $1 - p$ 로 발생한다.  
  >  - 예시:  
  >
  >    $$
  >    p(Y = \{1,0,1,0,0\}) = p(1-p)p(1-p)(1-p) = p^2(1-p)^3
  >    $$
  - 데이터셋 전체에 대한 우도(Likelihood)는 다음과 같이 표현된다:  

    $$
    \prod_{n=1}^{N} \hat{y}_n^{y_n} (1 - \hat{y}_n)^{1 - y_n}, \quad 
    \text{where } \hat{y}_n = p(y_n = 1 \mid x_n)
    $$

  - 우도(Likelihood) = 모델 하에서 데이터셋(레이블)이 관측될 확률  

---

## p20. 분류: 손실 함수 (이진)

- 우리는 다음을 가진다  

  $$
  \mathcal{D} = \{(x_n, y_n) \mid n = 1, \ldots, N\}, \quad \text{where } y_i \in \{0, 1\}
  $$

- 손실 유도(Loss derivation):  
  - 데이터셋 전체에 대한 우도(Likelihood)는 다음과 같이 표현된다:  

    $$
    \prod_{n=1}^{N} \hat{y}_n^{y_n} (1 - \hat{y}_n)^{1 - y_n}, 
    \quad \text{where } \hat{y}_n = p(y_n = 1 \mid x_n)
    $$

    $\hat{y}_n$: 예측된 확률(sigmoid 출력)

  - 우리는 로그 우도(log-likelihood)를 최대화(maximizing)하여 모델을 학습한다:  

    $$
    \mathcal{L}_{\text{BCE}}(\theta) 
    = -\sum_{n=1}^{N} 
    \Big[ y_n \log \hat{y}_n 
    + (1 - y_n)\log(1 - \hat{y}_n) \Big]
    $$

    - 이 손실 함수는 이진 교차 엔트로피(Binary Cross-Entropy, BCE)라고 불린다.  

---

## p21. 분류: 손실 함수 (다중 클래스)

<a href="https://en.wikipedia.org/wiki/Categorical_distribution" target="_blank">https://en.wikipedia.org/wiki/Categorical_distribution</a>

- 우리는 다음을 가진다  

  $$
  \mathcal{D} = \{(x_n, y_n) \mid n = 1, \ldots, N\}, \quad \text{where } y_i \in \{1, \ldots, C\}
  $$

- 손실 유도(Loss derivation):  
  - 각 레이블은 범주형 확률변수(Categorical random variable)로 모델링될 수 있다.  

    $$
    P(y_n = c \mid x_n) = \hat{y}_{n,c}, \quad 
    \sum_{c=1}^{C} \hat{y}_{n,c} = 1
    $$

    $\hat{y}_{n,c}$: 클래스 $c$ 에 대한 예측 확률(softmax 출력)

  - 데이터셋 전체에 대한 우도(Likelihood)는 다음과 같이 표현된다:  

    $$
    \prod_{n=1}^{N} \prod_{c=1}^{C} 
    \hat{y}_{n,c}^{\,1[y_n = c]}
    $$

  - 우리는 로그 우도(log-likelihood)를 최대화(maximizing)하여 모델을 학습한다:  

    $$
    \mathcal{L}_{\text{CE}}(\theta)
    = -\sum_{n=1}^{N} \sum_{c=1}^{C}
    1[y_n = c] \log \hat{y}_{n,c}
    $$

    - 이 손실 함수는 교차 엔트로피(Cross-Entropy, CE)라고 불린다.  

---

## p22. 이진 분류에서의 시그모이드 vs 소프트맥스

- 시그모이드(Sigmoid)와 소프트맥스(Softmax)는 서로 다른 가정에서 출발한다 (베르누이 vs. 범주형(Categorical)).  
- 그러나 이진 분류(binary case)에서는 분자와 분모를 나누면 동일한 형태가 된다.  
  - 실제로는 구현상의 선택(implementation choice)의 문제이다.  

**주어진 로짓(logits) $u_1, u_2$ 에 대해:**  

$$
p_1 = \frac{e^{u_1}}{e^{u_0} + e^{u_1}}
     = \frac{1}{1 + e^{u_0 - u_1}}
     = \sigma(u_1 - u_0)
$$

$$
p_0 = 
\underbrace{\frac{e^{u_0}}{e^{u_0} + e^{u_1}}}_{\text{Softmax}}
= 
\underbrace{\sigma(u_0 - u_1)}_{\text{Sigmoid}}
= 1 - p_1
$$

- 두 개의 로짓에 대한 소프트맥스는 로짓 차이 $(u_1 - u_0)$ 에 대한 시그모이드와 동일하다.  
- 따라서 이는 같은 문제를 푸는 조금 다른 관점일 뿐이다.  

**교차 엔트로피(CE, Softmax)는 이진 교차 엔트로피(BCE)로 축소된다:**  

$$
-\big[ y \log p_1 + (1 - y) \log(1 - p_1) \big]
$$

<img src="/assets/img/lecture/textmining/8/image_7.png" alt="image" width="600px">

---

## p23. 다중 레이블 분류

**다중 레이블 분류(Multi-label classification)**  
- 많은 실제(real-world) 사례에서, 각 인스턴스는 동시에 여러 클래스(multiple classes)에 속할 수 있다.  
- 예시: 텍스트 주제 분류(Text topic classification), 객체 탐지(Object detection)

<img src="/assets/img/lecture/textmining/8/image_8.png" alt="image" width="720px">

---

## p24. 다중 레이블 분류

>✓ 베르누이 분포(Bernoulli distribution):  
>  - 확률변수 $y = 1$ 은 확률 $p$ 로, $y = 0$ 은 확률 $1 - p$ 로 발생한다.  
>  - 예시:  
>
>    $$
>    p(Y = \{1,0,1,0,0\}) = p(1-p)p(1-p)(1-p) = p^2(1-p)^3
>    $$

- 손실 유도(Loss derivation):  
  - 각 클래스 레이블 $y_c$ 는 독립적인 베르누이 확률변수로 모델링될 수 있다.  

  $$
  P(y \mid x) = \prod_{c=1}^{C} 
  \hat{y}_c^{\,y_c} (1 - \hat{y}_c)^{\,1 - y_c}
  $$

  - 여기에서, $$\hat{y}_c = P(y_c = 1 \mid x) = \sigma(u_c)$$, 클래스 c에 대한 sigmoid 출력  

<img src="/assets/img/lecture/textmining/8/image_9.png" alt="image" width="800px">

---

## p25. 다중 레이블 분류

- 손실 유도(Loss derivation):  
  - 각 클래스 레이블 $y_c$ 는 독립적인 베르누이 확률변수로 모델링될 수 있다.  

  $$
  P(y \mid x) = \prod_{c=1}^{C} 
  \hat{y}_c^{\,y_c} (1 - \hat{y}_c)^{\,1 - y_c}
  $$

  - 여기에서, $$\hat{y}_c = P(y_c = 1 \mid x) = \sigma(u_c)$$, 클래스 c에 대한 sigmoid 출력  

- 음의 로그 우도(negative log-likelihood)는 BCE 손실(binary cross-entropy loss)로 표현되며,  
  각 클래스에 독립적으로 적용(applied independently per class)된다.  

  $$
  \mathcal{L}_{\text{Multi-label}}(\theta)
  = - \sum_{n=1}^{N} \sum_{c=1}^{C}
  \Big[ y_{n,c} \log \hat{y}_{n,c} + (1 - y_{n,c}) \log(1 - \hat{y}_{n,c}) \Big]
  $$

✓ 주의(Note):  
- 덜 일반적이지만(less common), 소프트맥스(Softmax)도 적용될 수 있다.  
  이 경우, 클래스 확률(class probabilities)은 반드시 합이 1이 되어야 하므로,  
  상호 배타적(mutually exclusive)인 레이블로 취급된다.  

---

### 보충 설명

#### 1. 다중 레이블 분류에서 소프트맥스(Softmax)의 제약  
- 다중 레이블 분류(Multi-label classification)는  
  하나의 샘플이 여러 클래스에 동시에 속할 수 있는 문제이다.  
  예를 들어, 한 이미지가 `cat=1`, `dog=0`, `human=1` 처럼  
  여러 레이블을 동시에 가질 수 있다.  
- 이때 일반적으로 시그모이드(Sigmoid) 함수를 사용한다.  
  각 클래스에 대해 독립적인 확률을 계산하므로,  
  확률의 합이 1이 될 필요가 없다.

#### 2. 소프트맥스(Softmax) 적용 시의 의미  
- 반면 소프트맥스(Softmax) 함수는  
  모든 클래스의 확률을 합했을 때 1이 되도록(normalized) 강제한다.  

  $$
  \sum_{c=1}^{C} p(y=c \mid x) = 1
  $$

- 따라서 소프트맥스를 사용하면 모델은  
  “하나의 샘플이 단 하나의 클래스에만 속한다”는  
  상호 배타적(mutually exclusive) 가정을 따르게 된다.  
- 즉, 소프트맥스는 다중 클래스 분류(Multi-class classification)에 적합하며,  
  다중 레이블 분류(Multi-label classification)에는 부적절하다.  

---

## p26. 분류 과제(Classification task): 요약

- 분류 유형과 손실:  
  - 이진 분류(Binary classification)  

    $$
    \mathcal{L}_{\text{BCE}}(\theta)
    = -\sum_{n=1}^{N}
    \big[ y_n \log \hat{y}_n + (1 - y_n)\log(1 - \hat{y}_n) \big]
    $$

  - 다중 클래스 분류(Multi-class classification)  

    $$
    \mathcal{L}_{\text{CE}}(\theta)
    = -\sum_{n=1}^{N} \sum_{c=1}^{C}
    1[y_n = c] \log \hat{y}_{n,c}
    $$

  - 다중 레이블 분류(Multi-label classification)  

    $$
    \mathcal{L}_{\text{Multi-label}}(\theta)
    = -\sum_{n=1}^{N} \sum_{c=1}^{C}
    \big[ y_{n,c}\log \hat{y}_{n,c} + (1 - y_{n,c})\log(1 - \hat{y}_{n,c}) \big]
    $$

- 전체 흐름을 생각하라!:  
  - 각 입력 텍스트는 사전학습된 모델(pretrained model)을 이용하여 벡터(vector)로 표현된다.  
  - 손실(loss)을 최소화함으로써 모델을 미세조정(fine-tuning)한다.  
    (전체 파라미터를 조정할 수도 있고, 일부만 조정할 수도 있다 — full vs. partial fine-tuning)

<img src="/assets/img/lecture/textmining/8/image_10.png" alt="image" width="800px">

---

# p27. 분류: 평가(evaluation)

---

## p28. 분류: 평가

- 우리의 분류기가 얼마나 잘 작동하는가?  

- 먼저 이진 분류기(binary classifiers)부터 살펴보자:  
  - 이 이메일은 스팸인가?  
    **→ spam (+) 또는 not spam (−)**  
    <img src="/assets/img/lecture/textmining/8/image_11.png" alt="image" width="480px">
  - 이 게시글은 Pie 회사에 관한 것인가?  
    **→ about pie (+) 또는 not about pie (−)**  

- 우리가 알아야 할 것들:  
  1. 분류기가 각 이메일 또는 게시글에 대해 **무엇을 예측했는가?**  
  2. 분류기가 **무엇을 예측했어야 하는가?**   
     - 정답은 **gold label** 또는 **ground-truth** 라고 불린다.  
     - 일반적으로 사람(human)에 의해 주석(annotated)된다.  


