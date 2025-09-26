---
layout: post
title: "[텍스트 마이닝] 4. Machine Learning 3"
date: 2025-09-26 12:00:00 +0900
categories:
  - "대학원 수업"
  - "텍스트 마이닝"
tags: []
---

> 출처: 텍스트 마이닝 – 강성구 교수님, 고려대학교 (2025)

## p2. 질문  

- **머신러닝의 진정한 목적은 무엇인가?**  
  1. 학습 세트(training set)에서의 오류를 최소화한다  
  2. 보이지 않는 미래의 예시들(unseen future examples)에서 오류를 최소화한다  
  3. 기계에 대해 배운다 (learn about machines)  
  4. 경사하강법(gradient descent)을 통해 학습 손실을 최소화한다  

---

# 일반화(Generalization)

---

## p5. 과소적합과 과대적합  

- **모델의 크기와 깊이를 늘리면 항상 일반화(generalization)가 좋아질까?**  
  - **아니다! (No!)**  

---

**과소적합 (Underfitting)**   

- 모델이 **너무 단순**해서 패턴을 포착하지 못한다.  
- **높은 학습 오류 (High training error)**  
- **높은 테스트 오류 (High test error)**   

<img src="/assets/img/textmining/4/image_1.png" alt="image" width="270px">

---

**좋은 일반화 (Good generalization)**  

- **낮은 학습 오류 (Low training error)**  
- **낮은 테스트 오류 (Low test error)**  

<img src="/assets/img/textmining/4/image_2.png" alt="image" width="270px">

---

**과대적합 (Overfitting)**  

- 모델이 학습 집합의 **모든 잡음(noise)** 을 외워버린다.  
- **낮은 학습 오류 (Low training error)**  
- **높은 테스트 오류 (High test error)**  

<img src="/assets/img/textmining/4/image_3.png" alt="image" width="270px">

---

## p6. 과소적합과 과대적합  

- **과소적합** 은 모델이 데이터 속의 **내재된 패턴(underlying patterns)** 을 포착하기에 **너무 단순할 때** 발생한다.  
- **과대적합** 은 모델이 일반화 가능한 패턴을 학습하지 않고, 학습 데이터를 **외워버릴 때(memorizes the training data)** 발생한다.  

---

<img src="/assets/img/textmining/4/image_4.png" alt="image" width="480px">

- 파란색: 훈련 세트(Training set)  
- 주황색: 테스트 세트(Test set)  
- 가로축: 모델 복잡도(가설 집합)  
- 세로축: 오류(Error)  

**참고:** 가설 집합(Hypothesis class)은 가능한 모든 예측기들의 집합이다.  

---

<img src="/assets/img/textmining/4/image_5.png" alt="image" width="600px"> 

---

### 보충 설명  

#### 1. **과소적합 구간**  
- 모델 복잡도가 너무 낮으면 데이터의 근본적인 패턴을 포착하지 못한다.  
- 이때 학습 오류와 테스트 오류가 모두 높게 나타난다.  

#### 2. **좋은 일반화 구간**  
- 적절한 수준의 모델 복잡도에서는 학습 데이터와 테스트 데이터 모두에서 오류가 낮아진다.  
- 이 지점이 일반화 성능이 가장 좋은 영역이다.  

#### 3. **과대적합 구간**  
- 모델 복잡도가 지나치게 높으면 학습 데이터의 잡음까지 외워버린다.  
- 그 결과 학습 오류는 낮지만 테스트 오류는 다시 높아진다.  
- 이는 모델이 새로운 데이터에 일반화하지 못하고, 훈련 데이터에만 특화된 복잡한 규칙을 학습했기 때문이다.  

---

## p7. 학습(fitting)과 일반화(generalization) 이해하기  

- 가능한 모든 예측기(predictors)의 공간을 생각해보자.  
  - 그 안에는 완벽하게 예측하는 **최적 예측기(optimal predictor)** $f^*$ 가 존재한다.  
  - 물론, 이 예측기는 도달할 수 없는(unattainable) 것이다.  
  - 그렇다면, 우리는 $f^*$ 로부터 얼마나 떨어져 있을까?  

<img src="/assets/img/textmining/4/image_6.png" alt="image" width="600px"> 

---

## p8. 학습(fitting)과 일반화(generalization) 이해하기  

- 실제로 우리는 가능한 모든 예측기(predictors)를 고려할 수 없다.  
  - 대신, **가설 집합(hypothesis class) $\mathcal{F}$** 을 정의하여, 공간을 관리 가능한 모델 집합으로 제한한다.  
  - 학습이 끝난 후, 우리는 **학습된 예측기(learned predictor) $\hat{f}$** 를 얻게 된다.  

<img src="/assets/img/textmining/4/image_7.png" alt="image" width="600px"> 

---

## p9. 학습(fitting)과 일반화(generalization) 이해하기  

- 실제로 우리는 가능한 모든 예측기(predictors)를 고려할 수 없다.  
  - 대신, **가설 집합(hypothesis class) $\mathcal{F}$** 을 정의하여, 공간을 관리 가능한 모델 집합으로 제한한다.  
  - 또한, $\mathcal{F}$ 안에는 **가능한 한 최선의 함수(the best possible function) $g$** 가 존재한다.   

<img src="/assets/img/textmining/4/image_8.png" alt="image" width="600px"> 

---

## p10. 학습(fitting)과 일반화(generalization) 이해하기  

- 우리는 $f^*$ 로부터 얼마나 떨어져 있는가?  

<img src="/assets/img/textmining/4/image_9.png" alt="image" width="600px">  

$$
\text{Err}(\hat{f}) - \text{Err}(f^*) 
= \underbrace{\text{Err}(\hat{f}) - \text{Err}(g)}_{\text{estimation error}} 
+ \underbrace{\text{Err}(g) - \text{Err}(f^*)}_{\text{approximation error}}
$$  

- **추정 오차(Estimation error)** 는 **제한된 데이터와 학습의 비효율성(limited data and learning inefficiencies)** 으로부터 발생한다.  
- **근사 오차(Approximation error)** 는 **가설 집합(hypothesis class)의 한계(limitations)** 으로부터 발생한다.  

---

## p11. 학습(fitting)과 일반화(generalization) 이해하기  

- 우리는 $f^*$ 로부터 얼마나 떨어져 있는가?  

<img src="/assets/img/textmining/4/image_10.png" alt="image" width="600px">  

- 가설 집합(hypothesis class)의 크기가 커질수록 (예: 선형 모델에서 심층 신경망으로 갈수록):  
  - **근사 오차(Approximation error)** 는 감소한다.  
  - 그 이유는 모델의 표현력이 더 풍부해져서, 최적 함수에 한층 더 가까워질 수 있기 때문이다.   

---

## p12. 학습(fitting)과 일반화(generalization) 이해하기  

- 우리는 $f^*$ 로부터 얼마나 떨어져 있는가?  

<img src="/assets/img/textmining/4/image_11.png" alt="image" width="600px">  

- 가설 집합(hypothesis class)의 크기가 커질수록 (예: 선형 모델에서 심층 신경망으로 갈수록):  
  - **근사 오차(Approximation error)** 는 감소한다.  
    - 모델이 더 표현력이 커져서(optimal function에 더 가까워질 수 있게 되어) 최적 함수에 접근할 수 있기 때문이다.  
  - **추정 오차(Estimation error)** 는 증가한다.  
    - 더 복잡한 가설 집합은 효과적으로 학습하기 위해 더 많은 데이터가 필요하기 때문이다.  
    - 데이터가 제한되어 있을 때, 표현력이 큰 모델은 일반화하지 못하고 학습 집합을 **외워버리는(overfitting)** 경향이 있다.  

---

## p13. 학습(fitting)과 일반화(generalization) 이해하기  

- 우리는 $f^*$ 로부터 얼마나 떨어져 있는가?  

<img src="/assets/img/textmining/4/image_12.png" alt="image" width="600px">  

- 가설 집합(hypothesis class)의 크기가 커질수록 (예: 선형 모델에서 심층 신경망으로 갈수록):  
  - **근사 오차(Approximation error)** 는 감소한다.  
    - 모델이 더 표현력이 풍부해져서 최적 함수에 더 가까워질 수 있기 때문이다.  
  - **추정 오차(Estimation error)** 는 증가한다.  
    - 더 복잡한 가설 집합은 효과적인 학습을 위해 더 많은 데이터가 필요하기 때문이다.  
    - 데이터가 제한되어 있을 때, 표현력이 큰 모델은 일반화하지 못하고 학습 집합을 **외워버리는(overfitting)** 경향이 있다.  

---

- 목표는 두 오차를 모두 최소화하는 **적절한 균형(right balance)** 을 찾아내어, 최상의 일반화 성능을 얻는 것이다.  

---

## p14. 어떻게 과대적합(overfitting)을 줄일 수 있을까?  

1. **특성 선택(Feature selection):** 입력 차원 줄이기  

   - 가능한 **모든 특성(all possible features)** 을 사용하는 대신,  
     **가장 관련 있는 특성들(only the most relevant ones)** 만 선택하여 복잡성을 줄일 수 있다.  

   - **왜 이것이 도움이 될까?**  
     - 더 많은 특성들은 모델 파라미터 수를 증가시킨다.  
     - 특성 선택은 모델을 단순화하면서, 필요한 정보만 유지할 수 있도록 해준다.  

---

## p15. 어떻게 과대적합(overfitting)을 줄일 수 있을까?  

1. **특성 선택(Feature selection):** 입력 차원 줄이기  

   - 가능한 **모든 특성(all possible features)** 을 사용하는 대신,  
     **가장 관련 있는 특성들(only the most relevant ones)** 만 선택하여 복잡성을 줄일 수 있다.  

   - **왜 이것이 도움이 될까?**  
     - 더 많은 특성들은 모델 파라미터 수를 증가시킨다.  
     - 특성 선택은 모델을 단순화하면서, 필요한 정보만 유지할 수 있도록 해준다.  

   - **어떻게 수행되는가(How is this done)?**  
     - 이 부분은 본 강의 범위를 벗어난다.  

---

<img src="/assets/img/textmining/4/image_13.png" alt="image" width="720px">  

- **특성 선택은 현재 활발히 연구되는 주제(hot research topic)이다.**  
- 최근 방법들은 **가장 관련 있는 입력 특성들(the most relevant input features)** 을 자동으로 찾아낸다.  

*(그림 출처: MvFS: Multi-view Feature Selection for Recommender System, ACM CIKM, 2023)*  
