---
layout: post
title: "[텍스트 마이닝] 3. Machine Learning 2"
date: 2025-09-24 09:40:00 +0900
categories:
  - "대학원 수업"
  - "텍스트 마이닝"
tags: []
---

> 출처: 텍스트 마이닝 – 강성구 교수님, 고려대학교 (2025)

# 비선형 특성(Non-linear features)

---

## p6. 선형 예측기와 비선형 특성  

- **질문(Q):** 선형 분류기를 사용하여 원형인 결정 경계를 얻을 수 있을까?  
  - **네! (Yes!)**  

  - **주의:** ‘선형(linear)’은 **가중치 벡터**와 **예측값** 사이의 관계를 의미한다. (**입력 $x$가 아님**)  

<img src="/assets/img/textmining/3/image_1.png" alt="image" width="600px">

---

## p7. 선형 예측기에서의 특성  

- 선형 모델의 경우, **가설 클래스**는 특성 추출기(feature extractor) $\varphi(x)$의 선택에 의해 결정된다.  

$$
\mathbf{w} \cdot \varphi(x) = \sum_{j=1}^{d} w_j \varphi(x)_j
$$  

- 따라서, 특성 설계는 데이터에서 의미 있는 관계를 포착하는 데 매우 중요하다.  

- 현실 세계 데이터에서의 비선형성(non-linearity)의 어려움:  
  1. **비단조성(non-monotonicity):** 특성과 목표 변수 간의 관계는 항상 단조적으로 증가하거나 감소하지 않는다.  
  2. **포화(saturation):** 일부 특성은 특정 지점을 넘어가면 효과가 줄어든다.  
  3. **특성 간 상호작용(interactions between features):** 하나의 특성의 영향은 다른 특성의 존재에 따라 달라질 수 있으며, 이는 선형 모델이 자연스럽게 포착할 수 없다.  

---

### 보충 설명 

- **가설 클래스(Hypothesis class)**란 모델이 학습 과정에서 선택할 수 있는 함수들의 집합을 의미한다.  
  - 이는 “어떤 모양의 함수들 중에서 최적의 것을 찾을 것인가”를 미리 정해 놓은 틀이다.  

- 예시:  
  - **선형 회귀**:  

    $$
    \mathcal{F} = \{ f_{\mathbf{w}}(x) = \mathbf{w} \cdot x \}
    $$  

    - 입력 $x$에 대해 직선(혹은 초평면) 형태의 함수만 고려한다.  

  - **비선형 특성 변환**:  

    $$
    \mathcal{F} = \{ f_{\mathbf{w}}(x) = \mathbf{w} \cdot \varphi(x) \}
    $$  

    - $\varphi(x)$를 사용해 다항식, 곡선, 원형 경계 등 더 복잡한 형태를 포함시킬 수 있다.  

- 즉, 가설 클래스는  
  - 모델이 어떤 함수 형태를 학습할 수 있는가?  
  - 문제를 어떤 방식으로 수학적으로 표현할 것인가?  
  - 를 결정하는 **모델의 표현력 범위**라 할 수 있다.  

---

## p8. 선형 예측기의 특징  

- **예: 건강 예측하기**  
  - **입력(Input):** 환자 정보 $x$  
  - **출력(Output):** 건강 $y \in \mathbb{R}$ (값이 클수록 더 좋음)  
  - 의료 진단을 위한 특징들: 키, 몸무게, 체온, 혈압 등  



  - **직접적인 접근법(straightforward approach)** 은 이 특징들을 입력으로 직접 사용하는 것임:  

$$
\varphi(x) = [1, height(x), weight(x), temperature(x), blood\;pressure(x)]
$$  

$$
y = w_1 + w_2 \, height(x) + w_3 \, weight(x) + w_4 \, temperature(x) + w_5 \, blood\;pressure(x)
$$  



- **그러나, 이 접근법은 한계가 있으며, 우리는 다음에서 이를 살펴볼 것임.**  

---

## p9. 비단조성(Non-monotonicity)   

- **도전 과제:** 특징들과 목표 변수 사이의 관계는 항상 단조적으로 증가하거나 감소하지 않는다.  

---

- **직접적인 접근법:**  

$$
\varphi(x) = [1, \; temperature(x)], \quad y = w_1 + w_2 \; temperature(x)
$$  

- **문제:** 모델은 극단적인 값을 선호하지만, 실제 관계는 비단조적이다. 

---

- **시도:** 이차 특징(quadratic features) 도입  

$$
\varphi(x) = [1, \; (temperature(x) - 36.5)^2], \quad y = w_1 + w_2 \; (temperature(x) - 36.5)^2
$$  

- **단점:** 이 접근법은 수동으로 도메인 특화(domain-specific) 변환을 정의해야 한다.  

---

### 보충 설명
  
1. **비단조성(non-monotonicity) 문제**  
- 선형 모델은 기본적으로 입력 특징(feature)과 출력 사이의 관계가 단조적(즉, 증가하거나 감소만 하는 경우)이라고 가정한다.  
- 하지만 실제 데이터에서는 온도와 건강 상태의 관계처럼 특정 구간에서는 증가하다가 이후에는 감소하는 **비단조적 패턴**이 자주 나타난다.  

2. **직접적 접근법의 한계**  
- 단순히 `temperature(x)`를 선형항으로 사용하는 경우, 모델은 "온도가 높을수록 건강이 좋다" 또는 "온도가 낮을수록 건강이 좋다" 중 하나만 학습한다.  
- 실제로는 정상 체온 근처에서 건강이 최적이고, 그보다 낮거나 높으면 건강이 나빠지는 **곡선 관계**를 포착하지 못한다.  

3. **이차 특징(quadratic feature)의 도입**  
- `(temperature(x) - 36.5)^2`와 같은 변환을 추가하면, 온도가 36.5에서 멀어질수록 건강이 나빠진다는 곡선 관계를 표현할 수 있다.  
- 이렇게 하면 선형 모델도 비단조적인 관계를 근사할 수 있다.  

4. **단점**  
- 이 방법은 **도메인 지식(domain knowledge)** 에 기반하여 사람이 직접 변환식을 정의해야 한다.  
- 복잡한 문제에서는 어떤 변환을 정의해야 할지 명확하지 않고, 모든 경우를 수동으로 설계하기 어렵다.  

---

## p10. 비단조성(Non-monotonicity)  
- **더 나은 접근법:**  
  결합할 수 있는 간단한 구성 요소(building blocks)로 특징을 설계한다.  

$$
\varphi(x) = [1, \; temperature(x), \; temperature(x)^2]
$$  

$$
y = w_1 + w_2 \; temperature(x) + w_3 \; temperature(x)^2
$$  

- 새로운 가설 클래스(hypothesis class)는 이전 것을 포괄하면서도 개념적으로 더 유연하다.  

> 💡 여기서 $w_1, w_2, w_3$는 사용자가 직접 정하는 값이 아니라,  
> **학습 과정(training)** 을 통해 데이터로부터 자동으로 추정된다.  

- **일반적인 규칙:**  
  복잡한 특징을 직접 설계하는 대신, 먼저 원하는 함수 형태를 고려한 후  
  그것을 간단한 구성 요소(building blocks)로 분해한다.  

---

## p11. 포화(Saturation)  

- **문제점:** 일부 특징들은 일정 지점을 넘어서면 효과가 감소한다.  

- **예시: 제품 추천**  
  - 입력: 제품 정보 $x$  
  - 출력: 관련성 $y \in \mathbb{R}$  
  - 특징 $N(x)$: 제품 $x$를 구매한 사람 수  

- **직접적인 접근법:**  

$$
\varphi(x) = [1, N(x)], \quad y = w_1 + w_2 N(x)
$$  

- **문제:** 1000명이 구매한 제품에 비해 10000명이 구매한 제품이 정말로 10배의 관련성을 갖는가?  
  - 그렇지 않다! 인기는 포화될 수 있으며, 추가적인 구매는 인지된 관련성에 덜 기여할 수 있다.  

---

## p13. 특성들 간의 상호작용 

- **예시: 건강 예측**  
  - 입력: 환자 정보 $x$  
  - 출력: 건강 $y \in \mathbb{R}$ (높을수록 더 좋음)  
  - 두 특징을 고려: 키(height)와 몸무게(weight).  
    건강한 몸무게 범위는 명확히 키에 의존한다.  
---

- **직접적인 접근법:** 이 관계를 포착할 수 없다.  

$$
\varphi(x) = [1, \; height(x), \; weight(x)], \quad 
y = w_1 + w_2 \; height(x) + w_3 \; weight(x)
$$  

---

- **더 나은 접근법:** 여러 상호작용 항을 포함하는 특징들을 추가  

$$
\varphi(x) = [1, \; height(x), \; weight(x), \; height(x)weight(x), \; weight(x)/height(x)^2]
$$  

$$
y = w_1 + w_2 \; height(x) + w_3 \; weight(x) + w_4 \; height(x)weight(x) + w_5 \; weight(x)/height(x)^2
$$  

---

- $BMI = \dfrac{weight(kg)}{height(m)^2}$  
  - 도메인 지식(domain knowledge)이 반영될 수 있다.  

---

## p14. 비선형 특성 설계: 회귀 

- **이차 특성(Quadratic features)**  

$$
\varphi(x) = [1, x, x^2]
$$  

$$
f(x) = [2, 1, -0.2] \cdot \varphi(x)
$$  

$$
f(x) = [4, -1, 0.1] \cdot \varphi(x)
$$  

$$
f(x) = [1, 1, 0] \cdot \varphi(x)
$$  

$$
\mathcal{F} = \{ f_{\mathbf{w}}(x) = \mathbf{w} \cdot \varphi(x) : \mathbf{w} \in \mathbb{R}^3 \}
$$  

<img src="/assets/img/textmining/3/image_2.png" alt="image" width="480px">

---

- **주기성 특성(Periodicity features)**  

$$
\varphi(x) = [1, x, x^2, \cos(3x)]
$$  

$$
f(x) = [1, 1, -0.1, 1] \cdot \varphi(x)
$$  

$$
f(x) = [3, -1, 0.1, 0.5] \cdot \varphi(x)
$$  

$$
\mathcal{F} = \{ f_{\mathbf{w}}(x) = \mathbf{w} \cdot \varphi(x) : \mathbf{w} \in \mathbb{R}^4 \}
$$  

<img src="/assets/img/textmining/3/image_3.png" alt="image" width="480px">