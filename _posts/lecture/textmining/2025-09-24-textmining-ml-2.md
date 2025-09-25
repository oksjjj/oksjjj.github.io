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
  를 결정하는 **모델의 표현력 범위**라 할 수 있다.  

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

## p12. 포화(Saturation)의 개선 방법  

- **더 나은 접근법들:**  

1. **로그 변환(Logarithmic transformation)**  

   $$
   \varphi(x) = [1, \log N(x)], \quad y = w_1 + w_2 \log N(x)
   $$  

   - 값의 범위(dynamic range)가 매우 클 때 좋은 아이디어가 될 수 있다.  

2. **이산화(Discretization, binning)**  

   $$
   \varphi(x) = [1, \; 1[0 < N(x) \leq 10], \; 1[10 < N(x) \leq 50], \; \dots, \; 1[500 < N(x) \leq 1000]]
   $$  

   $$
   y = w_1 + w_2 \; 1[0 < N(x) \leq 10] + \cdots + w_6 \; 1[500 < N(x) \leq 1000]
   $$  

   - $N(x)$는 미리 정의된 구간(bin)으로 나누어진다.  
   - 각 구간마다 다른 가중치를 학습할 수 있어, 구간별로 상이한 영향을 반영할 수 있다.  

---

## p13. 특성들 간의 상호작용 

- **예시: 건강 예측**  
  - 입력: 환자 정보 $x$  
  - 출력: 건강 $y \in \mathbb{R}$ (높을수록 더 좋음)  
  - 두 특징을 고려: 키(height)와 몸무게(weight).  
    건강한 몸무게 범위는 명확히 키에 의존한다.  

- **직접적인 접근법:** 이 관계를 포착할 수 없다.  

$$
\varphi(x) = [1, \; height(x), \; weight(x)], \quad 
y = w_1 + w_2 \; height(x) + w_3 \; weight(x)
$$  

- **더 나은 접근법:** 여러 상호작용 항을 포함하는 특징들을 추가  

$$
\varphi(x) = [1, \; height(x), \; weight(x), \; height(x)weight(x), \; weight(x)/height(x)^2]
$$  

$$
y = w_1 + w_2 \; height(x) + w_3 \; weight(x) + w_4 \; height(x)weight(x) + w_5 \; weight(x)/height(x)^2
$$  

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

---

### 보충 설명

#### 1. **이차 특성의 예시 전개**

- 특성 벡터:  
  $$
  \varphi(x) = [1, x, x^2]
  $$  

- 첫 번째 경우:  
  $$
  f(x) = [2, 1, -0.2] \cdot [1, x, x^2] = 2 + x - 0.2x^2
  $$  

- 두 번째 경우:  
  $$
  f(x) = [4, -1, 0.1] \cdot [1, x, x^2] = 4 - x + 0.1x^2
  $$  

- 세 번째 경우:  
  $$
  f(x) = [1, 1, 0] \cdot [1, x, x^2] = 1 + x
  $$  

---

#### 2. **주기성 특성의 예시 전개**

- 특성 벡터:  
  $$
  \varphi(x) = [1, x, x^2, \cos(3x)]
  $$  

- 첫 번째 경우:  
  $$
  f(x) = [1, 1, -0.1, 1] \cdot [1, x, x^2, \cos(3x)] = 1 + x - 0.1x^2 + \cos(3x)
  $$  

- 두 번째 경우:  
  $$
  f(x) = [3, -1, 0.1, 0.5] \cdot [1, x, x^2, \cos(3x)] = 3 - x + 0.1x^2 + 0.5\cos(3x)
  $$  

---

## p15. 비선형 특성 설계: 분류  

- **이차 특성(Quadratic features)**  

$$
\varphi(x) = [x_1, x_2, x_1^2 + x_2^2]
$$  

$$
f(x) = \text{sign}([2, 2, -1] \cdot \varphi(x))
$$  

$$
\mathcal{F} = \{ f_{\mathbf{w}}(x) = \mathbf{w} \cdot \varphi(x) : \mathbf{w} \in \mathbb{R}^3 \}
$$  

---

- **등가적 표현(Equivalently):**

$$
f(x) =
\begin{cases}
1 & \text{if } (x_1 - 1)^2 + (x_2 - 1)^2 \leq 2 \\
-1 & \text{otherwise}
\end{cases}
$$  

- 원래 입력 공간에서, 결정 경계(decision boundary)는 **원(circle)** 이다.  
- 변환된 특성 공간에서는, 결정 경계가 **초평면(hyperplane)** 이 된다.  

<img src="/assets/img/textmining/3/image_4.png" alt="image" width="600px">

---

### 보충 설명

#### 1. 원래 공간에서의 결정 경계  
- 입력이 $(x_1, x_2)$일 때 결정 함수는  

  $$
  f(x) = 2x_1 + 2x_2 - (x_1^2 + x_2^2)
  $$  

  이다.  
- $f(x) = 0$으로 두면  

  $$
  (x_1 - 1)^2 + (x_2 - 1)^2 = 2
  $$  

  가 되어 원래의 공간에서는 **원의 형태**가 결정 경계가 된다.  

#### 2. 변환된 공간에서의 표현  
- 특성 벡터를  

  $$
  \varphi(x) = [x_1, x_2, x_1^2 + x_2^2]
  $$  

  로 정의한다.  
- 새로운 좌표 $(z_1, z_2, z_3)$를  

  $$
  z_1 = x_1, \quad z_2 = x_2, \quad z_3 = x_1^2 + x_2^2
  $$  

  라고 두면, 결정 함수는  

  $$
  f(x) = [2, 2, -1] \cdot \varphi(x) = 2z_1 + 2z_2 - z_3
  $$  

  로 표현된다.  

#### 3. 초평면으로 단순화되는 이유  
- 결정 경계 $f(x) = 0$은  

  $$
  2z_1 + 2z_2 - z_3 = 0
  $$  

  의 형태가 된다.  
- 이는 $(z_1, z_2, z_3)$ 공간에서의 **평면 방정식**이며, 따라서 변환된 공간에서는 단순한 **선형 초평면**으로 표현된다.  

---

## p16. 선형 예측기와 비선형 특성  

- **무엇에 대해 선형인가?**  

  - 예측은 점수(score)에 의해 결정된다: $$\mathbf{w} \cdot \varphi(x) = \sum_{j=1}^d w_j \varphi(x)_j$$  

  - $\mathbf{w}$에 대해 선형인가? → **예 (Yes)**  
  - $\varphi(x)$에 대해 선형인가? → **예 (Yes)**  
  - $x$에 대해 선형인가? → **아니오 (No!)**  
    ($x$는 반드시 벡터일 필요조차 없다)

---

- **요약 (Summary):**

1. 선형 예측기 $f_{\mathbf{w}}(x)$는 비선형 함수(non-linear functions)를 모델링할 수 있으며, $x$에 대한 비선형 결정 경계(non-linear decision boundaries)를 만들 수 있다.  
2. 점수(score) $\mathbf{w} \cdot \varphi(x)$는 $\mathbf{w}$에 대한 선형 함수(linear function)이므로 효율적인 학습이 가능하다.  
3. 선형 예측기(linear predictors)는 도메인 지식(domain knowledge)에 기반한 잘 설계된 특성과 결합될 때 여전히 매우 효과적이다.  

---

# 신경망(Neural networks)

---

## p18. 비선형 예측기  

- **선형 예측기**  

$$
f_{\mathbf{w}}(x) = \mathbf{w} \cdot \varphi(x), \quad \varphi(x) = [1, x]
$$  

---

- **비선형 특성을 가진 선형 예측기**  
  - $\varphi(x)$를 바꾸어 비선형 함수를 모델링할 수 있다.  

$$
f_{\mathbf{w}}(x) = \mathbf{w} \cdot \varphi(x), \quad \varphi(x) = [1, x, x^2]
$$  

---

- **비선형 신경망**  
  - $\varphi(x)$를 사람이 직접 설계하는 대신, 신경망을 이용해 복잡한 변환(complex transformations)을 자동으로 학습할 수 있다.  

$$
f_{\mathbf{w}}(x) = \mathbf{w} \cdot \sigma(\mathbf{V}\varphi(x)), \quad \varphi(x) = [1, x]
$$  
  
> 최적의 transformation을 직접 디자인하는 것은 어려움  
> $\sigma(\mathbf{V}\varphi(x))$는 transformation을 자동으로 찾는 과정

---

## p19. 동기 부여 예시  

- **예시: 자동차 충돌 예측 (car collision prediction)**  
  - 입력: 마주 오는 두 자동차의 위치  

    $$
    x = [x_1, x_2]
    $$  

  - 출력: 안전(safe) 여부 또는 충돌(collide) 여부  

    $$
    y = +1 \; \text{(안전)}, \quad y = -1 \; \text{(충돌)}
    $$  

<img src="/assets/img/textmining/3/image_5.png" alt="image" width="240px">

- **참 함수(true function)를 가정:**  
  - 두 자동차가 충분히 멀리 떨어져 있으면(거리 ≥ 1) 안전하다.  

  $$
  y = \text{sign}(|x_1 - x_2| - 1)
  $$  

<img src="/assets/img/textmining/3/image_6.png" alt="image" width="480px">

---

## p20. 문제를 분해하기  

- 신경망을 이해하는 한 가지 방법은(뇌를 예로 들지 않고도) **문제 분해(problem decomposition)** 이다.  

- **[하위 문제 1]** 자동차 1이 자동차 2의 오른쪽에 충분히 멀리 있는지 확인:  

  $$
  h_1(x) = 1[x_1 - x_2 \geq 1]
  $$  

- **[하위 문제 2]** 자동차 2가 자동차 1의 오른쪽에 충분히 멀리 있는지 확인:  

  $$
  h_2(x) = 1[x_2 - x_1 \geq 1]
  $$  

<img src="/assets/img/textmining/3/image_7.png" alt="image" width="240px">

- **[예측]** 둘 중 하나라도 참이면 안전(safe):  

  $$
  f(x) = \text{sign}(h_1(x) - h_2(x))
  $$  

<img src="/assets/img/textmining/3/image_8.png" alt="image" width="240px">

---

## p21. 벡터 표기를 사용한 재작성  

- **특성 벡터**:  

  $$
  \varphi(x) = [1, x_1, x_2]
  $$  

- **중간 하위 문제들:**

$$
h_1(x) = 1[x_1 - x_2 \geq 1] = 1[\mathbf{v}_1 \cdot \varphi(x) \geq 0], \quad \mathbf{v}_1 = [-1, +1, -1]
$$  

$$
h_2(x) = 1[x_2 - x_1 \geq 1] = 1[\mathbf{v}_2 \cdot \varphi(x) \geq 0], \quad \mathbf{v}_2 = [-1, -1, +1]
$$  

> $1[\text{조건}]$ 는 조건이 참이면 1, 거짓이면 0을 반환하는 Indicator 함수이다.

- **최종 예측:**

$$
f_{\mathbf{v}, \mathbf{w}}(x) = \text{sign}(w_1 h_1(x) + w_2 h_2(x))
$$  

- **$\varphi(x)$가 주어졌을 때 우리의 목표는 아래 2개를 학습하는 것이다**  
  1. 숨겨진 하위 문제들 $\mathbf{V} = (\mathbf{v}_1, \mathbf{v}_2)$  
  2. 결합 가중치 $\mathbf{w} = [w_1, w_2]$  

---

## p22. Zero 그래디언트 피하기  

- 우리는 학습을 위해 경사하강법(gradient descent)을 사용하지만, 중요한 문제가 발생한다:  

  $$
  h(x) = 1[\mathbf{v} \cdot \varphi(x) \geq 0]
  $$  

    에서는 $\mathbf{v}$에 대한 $h(x)$의 그래디언트가 거의 모든 구간에서 0이 된다.  

- **해결책:**  
  - 0이 아닌 그래디언트를 보장하기 위해 매끄러운 활성화 함수(smooth activation function) $\sigma$로 대체한다.  

  $$
  h(x) = \sigma(\mathbf{v} \cdot \varphi(x))
  $$  

- **$\sigma$로 주로 활용되는 함수들**  
  - Threshold:  

    $$
    1[z \geq 0]
    $$  

  - Logistic (Sigmoid):  

    $$
    \frac{1}{1+e^{-z}}
    $$  

  - ReLU (rectified linear unit):  

    $$
    \max(z, 0)
    $$  

  - 기타 변형들: Leaky ReLU, ELU, SELU, GELU 등  

  <img src="/assets/img/textmining/3/image_9.png" alt="image" width="480px">

---

### 보충 설명  

#### 1. **Indicator 함수의 불연속성**  
- 함수 $$h(x) = 1[\mathbf{v} \cdot \varphi(x) \geq 0]$$은 $\mathbf{v} \cdot \varphi(x)$가 0을 기준으로 값이 갑자기 0에서 1로 바뀌는 **불연속 함수**이다.  
- 즉, 대부분의 영역에서는 값이 일정하게 유지되고, 기준점에서만 점프(discontinuity)가 발생한다.  

#### 2. **거의 모든 구간에서 그래디언트가 0인 이유**  
- 함수 값이 일정하게 유지되는 구간에서는 변화율이 없으므로 그래디언트가 0이다.  
- 오직 $\mathbf{v} \cdot \varphi(x) = 0$인 경계점에서만 값이 불연속적으로 바뀌므로, 그 외의 거의 모든 지점에서 그래디언트는 0이다.  

#### 3. **학습 과정에서의 문제**  
- 경사하강법(gradient descent)은 그래디언트를 이용해 매개변수 $\mathbf{v}$를 갱신한다.  
- 그런데 그래디언트가 0이면 매개변수가 더 이상 업데이트되지 않으므로 학습이 진행되지 않는다.  
- 이 문제를 해결하기 위해 **매끄러운 활성화 함수(smooth activation function)** 를 사용해 비제로(non-zero) 그래디언트를 확보한다.  

---

## p23. 신경망 (Neural networks)  

- 이제 우리는 **2-계층(two-layer) 신경망**을 정의할 준비가 되었다:  

<img src="/assets/img/textmining/3/image_10.png" alt="image" width="600px">

- **중간(hidden) 유닛들:**  

  $$
  h_j = \sigma(\mathbf{v}_j \cdot \varphi(x))
  $$  

- **출력(Output):**  

  $$
  \text{score} = \mathbf{w} \cdot \mathbf{h}
  $$  

- **표기**  
  - $\mathbf{V}$ : 첫 번째 계층(first layer)의 가중치  
  - $\mathbf{w}$ : 두 번째 계층(second layer)의 가중치  

- **Key insight:**  
  - 중간(hidden) 유닛들은 **선형 예측기(linear predictor)** 의 학습된 특징(learned features)으로 작동한다.  

---

### 보충 설명  

#### 1. **학습된 특징(learned features)의 의미**  
- 전통적인 선형 예측기는 사람이 직접 설계한 특성(feature design)에 의존한다.  
- 그러나 신경망의 중간(hidden) 유닛들은 학습 과정에서 데이터로부터 유용한 패턴을 자동으로 추출한다.  
- 따라서 이 유닛들은 **스스로 학습된 특징(learned features)** 으로 해석될 수 있다.  

#### 2. **선형 예측기와의 관계**  
- 최종 출력은 여전히 선형 결합  

  $$
  \text{score} = \mathbf{w} \cdot \mathbf{h}
  $$  

  의 형태를 따른다.  
- 하지만 입력 $x$ 자체가 아니라, 중간 유닛들이 만들어낸 **변환된 표현 $\mathbf{h}$** 에 대해 선형 결합을 수행한다는 점에서, 단순한 선형 모델보다 훨씬 복잡한 관계를 표현할 수 있다.  

#### 3. **신경망에서의 입력 처리**  
- 신경망에서는 $\varphi(x)$에 대해 별도의 feature design(특성 설계)을 하지 않고, **주어진 입력을 그대로 사용**한다.  
- 복잡한 특징 변환은 네트워크 내부의 계층과 활성화 함수에 의해 자동으로 이루어지며, 이는 사람이 일일이 설계하지 않아도 된다는 장점을 제공한다.  

---

## p24. 신경망에서의 특성 학습  

- 중간(hidden) 유닛들은 선형 예측기의 학습된 **특성**으로 작동한다.  

- **선형 모델:**  

  - 점수(score)는 주어진 특성 $\varphi(x)$의 선형 결합으로 계산된다.  

  $$
  \text{score} = \mathbf{w} \cdot \varphi(x)
  $$  

  - 선형 예측기는 사람이 직접 지정한 특성 $\varphi(x)$에 적용된다.  

  <img src="/assets/img/textmining/3/image_11.png" alt="image" width="300px">


- **신경망:**  

  - 입력 $\varphi(x)$는 은닉층(hidden layer)을 거쳐 새로운 표현 $h(x)$로 변환된다.  
  - 점수(score)는 이 학습된 표현을 기반으로 계산된다.  

  $$
  \text{score} = \mathbf{w} \cdot \mathbf{h}
  $$  

  - 선형 예측기는 사람이 지정한 특성이 아니라, **자동으로 학습된 특성**  

    $$
    h(x) = [h_1(x), \dots, h_k(x)]
    $$  

    에 적용된다.  

  <img src="/assets/img/textmining/3/image_12.png" alt="image" width="480px">

  ---

  ## p25. 심층 신경망  

- 우리는 이러한 개념들을 확장하여 더 깊은 신경망을 구축할 수 있다.  

- **1-계층 신경망 (선형 예측기, linear predictor):**  

  $$
  \text{score} = \mathbf{w} \cdot \varphi(x)
  $$  
  - 은닉층(hidden layer)의 수: 0  

  <img src="/assets/img/textmining/3/image_13.png" alt="image" width="220px">

- **2-계층 신경망:**  

  $$
  \text{score} = \mathbf{w} \cdot \sigma(\mathbf{V} \varphi(x))
  $$  
  - 은닉층의 수: 1  

  <img src="/assets/img/textmining/3/image_14.png" alt="image" width="300px">

- **3-계층 신경망:**  

  $$
  \text{score} = \mathbf{w} \cdot \sigma(\mathbf{V}_2 \, \sigma(\mathbf{V}_1 \varphi(x)))
  $$  
  - 은닉층의 수: 2  

  <img src="/assets/img/textmining/3/image_15.png" alt="image" width="380px">

---

### 보충 설명  

#### 1. **1-계층 신경망 (선형 예측기)**  
- 계산 과정:  

  $$
  \text{score} = \mathbf{w} \cdot \varphi(x)
  $$  

- 의미:  
  - 입력 $\varphi(x)$를 가중치 벡터 $\mathbf{w}$와 단순히 선형 결합한 값이다.  
  - 은닉층이 없으므로 비선형 변환은 일어나지 않는다.  
  - 기본적인 **선형 모델**에 해당한다.  

---

#### 2. **2-계층 신경망**  
- 계산 과정:  
  1. 은닉층 계산:  

     $$
     h = \sigma(\mathbf{V} \varphi(x))
     $$  

     - $\mathbf{V}$ : 입력을 은닉 유닛으로 변환하는 가중치 행렬  
     - $\sigma$ : 비선형 활성화 함수(예: sigmoid, ReLU)  
  2. 출력 계산:  

     $$
     \text{score} = \mathbf{w} \cdot h
     $$  

- 의미:  
  - $\mathbf{V}$와 $\sigma$에 의해 입력이 비선형적으로 변환된다.  
  - 변환된 표현 $h$는 사람이 설계하지 않은 **학습된 특성**으로 해석할 수 있다.  
  - $\mathbf{w}$는 이 학습된 특성들을 다시 선형 결합하여 최종 출력을 만든다.  

---

#### 3. **3-계층 신경망**  
- 계산 과정:  
  1. 첫 번째 은닉층:  

     $$
     h^{(1)} = \sigma(\mathbf{V}_1 \varphi(x))
     $$  
     
  2. 두 번째 은닉층:  

     $$
     h^{(2)} = \sigma(\mathbf{V}_2 h^{(1)})
     $$  

  3. 출력 계산:  

     $$
     \text{score} = \mathbf{w} \cdot h^{(2)}
     $$  

- 의미:  
  - 입력 $\varphi(x)$가 여러 번의 비선형 변환을 거치며 점점 더 복잡한 특성으로 추출된다.  
  - $\mathbf{V}_1, \mathbf{V}_2$는 각 층에서 입력을 새로운 표현으로 바꾸는 가중치 행렬이다.  
  - $\sigma$는 각 층에서 비선형성을 부여하여 단순한 선형 모델이 표현할 수 없는 복잡한 패턴을 학습할 수 있도록 한다.  
  - 마지막에 $\mathbf{w}$가 이 최종 표현을 결합하여 예측을 만든다.  

---

## p26. 왜 더 깊게 가는가?  

- 더 깊은 신경망은 단순한 특성들의 조합을 더 복잡한 패턴으로 확장할 수 있으며,  
  이는 더 나은 일반화(generalization)로 이어질 수 있다.  

---

- **원시 픽셀 (입력)**  
  - 원시 입력은 픽셀 값들로 구성된다.  

<img src="/assets/img/textmining/3/image_16.png" alt="image" width="50px">  

<p align="center">⬇️</p> 

- **1번째 계층**  
  - 하위 계층은 단순한 패턴을 감지한다.  
  - 예: 에지(edges)  

<img src="/assets/img/textmining/3/image_17.png" alt="image" width="100px">  

<p align="center">⬇️</p>  

- **2번째 계층**  
  - 에지들의 조합이 곡선, 질감, 그리고 객체의 부분을 형성한다.  
  - 예: 눈, 코, 입  

<img src="/assets/img/textmining/3/image_18.png" alt="image" width="200px">  

<p align="center">⬇️</p>  

- **3번째 계층**  
  - 더 높은 계층에서는 뉴런이 전체 객체에 반응한다.  
  - 예: 얼굴 전체  

<img src="/assets/img/textmining/3/image_19.png" alt="image" width="300px">  

---

> **점점 더 추상적이고 고수준의 특성을 표현할 수 있다**  

---

## p27. 왜 더 깊게 가는가?  

- 원칙적으로, **하나의 은닉층(one hidden layer)** 을 가진 신경망은 어떤 함수도 근사할 수 있다.  

  ✔ **보편 근사 정리(Universal approximation theorem) [1]**  
  - $\mathbb{R}^d$ 위의 임의의 연속 함수는, 유한한 수의 노드와 비선형 활성화 함수를 포함하는 하나의 은닉층 신경망에 의해 임의의 정밀도로 근사될 수 있다.  

- 그러나, **층의 크기(layer size)** 가 비현실적으로 커져야 하므로 학습이 매우 느려진다.  

---

- 더 깊은 신경망은 얕고 넓은 신경망과 동일한 함수들을 표현할 수 있다.  
- 하지만 **더 적은 매개변수(parameters)** 로 가능하며, 이는 더 나은 일반화(generalization)로 이어질 수 있다.  

---

<img src="/assets/img/textmining/3/image_20.png" alt="image" width="720px">  

---

### 보충 설명  

#### 1. **1개 은닉층 신경망 (총 75개 가중치)**  
- 입력 노드 = 4  
- 은닉 노드 = 15  
- 각 은닉 노드 = 입력 4개 + bias 1개 = 5  
- 총 가중치 수 = $15 \times 5 = 75$  

#### 2. **3개 은닉층 신경망 (총 75개 가중치)**  
- **첫 번째 은닉층 (bias 포함)**  
   - 입력 = 4  
   - 각 노드 = 입력 4개 + bias 1개 = 5  
   - 가중치 수 = $5 \times 5 = 25$  

- **두 번째 은닉층 (bias 제외)**  
   - 입력 = 첫 번째 은닉층 출력 5개  
   - 각 노드 = 입력 5개  
   - 가중치 수 = $5 \times 5 = 25$  

- **세 번째 은닉층 (bias 제외)**  
   - 입력 = 두 번째 은닉층 출력 5개  
   - 각 노드 = 입력 5개  
   - 가중치 수 = $5 \times 5 = 25$  

- 총합: $25 + 25 + 25 = 75$  

#### 3. **경로 수 비교 (입력 1개 기준)**  
- **1개 은닉층 신경망**  
  - 입력 1개가 은닉층의 15개 노드 각각으로 연결된다.  
  - 각 은닉 노드에서 출력으로 바로 연결 → 경로 수 = **15**  

- **3개 은닉층 신경망**  
  - 입력 1개가 첫 번째 은닉층의 5개 노드로 연결된다.  
  - 첫 번째 은닉층의 각 노드 → 두 번째 은닉층의 5개 노드로 확장  
  - 두 번째 은닉층의 각 노드 → 세 번째 은닉층의 5개 노드로 확장  
  - 최종적으로 출력과 연결됨  
  - 경로 수 = $5 \times 5 \times 5 = 125$  

#### 4. **의미**  
- 1층과 3층 모두 동일하게 **총 가중치 수 = 75개**이지만,  
- 3층 구조는 훨씬 많은 경로(125)를 가지므로 같은 가중치 수로 더 복잡한 표현을 학습할 수 있다.  
