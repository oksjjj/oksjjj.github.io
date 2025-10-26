---
layout: post
title: "[2. Variational Autoencoders] An Introduction to Variational Autoencoders"
date: 2025-10-23 13:00:00 +0900
categories:
  - "Books"
  - "An Introduction to Variational Autoencoders"
tags: []
---

# 2. Variational Autoencoders

이 장에서는 변분 오토인코더(VAE, Variational Autoencoder)의 기본 개념을 다룬다.  

---

## 2.1 인코더(Encoder) 또는 근사 사후분포(Approximate Posterior)

이전 장에서는 심층 잠재 변수 모델(DLVM, Deep Latent Variable Model)과  
그러한 모델에서 로그우도(log-likelihood)와 사후분포(posterior distribution)를  
추정하는 문제를 소개하였다.  

변분 오토인코더(VAE) 프레임워크는  
확률적 경사하강법(SGD, Stochastic Gradient Descent)을 이용하여  
DLVM과 이에 대응하는 추론 모델(inference model)을  
동시에 최적화할 수 있는 계산 효율적인 방법을 제공한다.

---

DLVM의 계산 불가능한(intractable) 사후 추론과 학습 문제를 계산 가능한 형태로 바꾸기 위해,  
매개변수를 갖는 추론 모델(parametric inference model) $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$를 도입한다.  

이 모델은 인코더(encoder) 또는 인식 모델(recognition model)이라고도 불린다.  
여기서 $\boldsymbol{\phi}$는 이 추론 모델의 매개변수(parameter)를 의미하며,  
변분 매개변수(variational parameters)라고 부른다.  

이 변분 매개변수 $\boldsymbol{\phi}$를 최적화하여 다음이 성립하도록 한다:

$$
q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \approx p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})
\tag{2.1}
$$

이후 설명하겠지만, 이러한 사후분포의 근사는  
주변우도(marginal likelihood)를 효율적으로 최적화하는 데 도움이 된다.

---

DLVM과 마찬가지로, 추론 모델(inference model) $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$은  
(거의) 모든 형태의 유향 그래픽 모델(directed graphical model)로 표현될 수 있다.

$$
q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})
= q_{\boldsymbol{\phi}}(\mathbf{z}_1, \ldots, \mathbf{z}_M \mid \mathbf{x})
= \prod_{j=1}^{M} q_{\boldsymbol{\phi}}(\mathbf{z}_j \mid Pa(\mathbf{z}_j), \mathbf{x})
\tag{2.2}
$$

여기서 $Pa(\mathbf{z}_j)$는 유향 그래프 내에서 변수 $\mathbf{z}_j$의 부모 변수 집합(parent variables)을 의미한다.  

DLVM과 유사하게, 분포 $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$는 심층 신경망을 사용하여 파라미터화될 수 있다.  
이 경우, 변분 매개변수 $\boldsymbol{\phi}$는 신경망의 가중치(weight)와 편향(bias)을 포함한다.  

예를 들어, 다음과 같이 표현할 수 있다:

$$
(\boldsymbol{\mu}, \log \boldsymbol{\sigma})
= \text{EncoderNeuralNet}_{\boldsymbol{\phi}}(\mathbf{x})
\tag{2.3}
$$

$$
q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})
= \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}))
\tag{2.4}
$$

---

일반적으로 VAE에서는 하나의 인코더 신경망을 사용하여  
데이터셋의 모든 데이터 포인트에 대한 사후 추론(posterior inference)을 수행한다.  

이는 전통적인 변분 추론(variational inference) 방법과 대조된다.  
기존 방법에서는 변분 매개변수(variational parameters)가 데이터 포인트별로 분리되어 있으며,  
각 데이터 포인트마다 별도의 반복적 최적화(iterative optimization)를 수행해야 한다.  

반면, VAE에서는 데이터 포인트 전체에 대해  
하나의 공통된 변분 매개변수를 공유하는 전략을 사용한다.  

이 방식을  
상각 변분 추론(amortized variational inference) (Gershman & Goodman, 2014)이라 부른다.  

상각 추론을 사용하면 데이터 포인트마다 개별적인 최적화 루프를 돌 필요가 없으며,  
확률적 경사하강법(SGD)의 효율성을 그대로 활용할 수 있다.

---

## 2.2 Evidence Lower Bound (ELBO)

변분 오토인코더(Variational Autoencoder, VAE)의 최적화 목적 함수는  
다른 변분 방법들과 마찬가지로 증거 하한(Evidence Lower Bound, ELBO)이다.  
이 목적 함수는 변분 하한(Variational Lower Bound)이라고도 불린다.  

일반적으로 ELBO는 젠슨 부등식(Jensen’s inequality)을 통해 유도된다.

여기서는 젠슨 부등식을 사용하지 않고  
ELBO가 얼마나 실제 값에 근접한지(tightness)를  
보다 직관적으로 이해할 수 있는 대안적 유도 방식을 사용할 것이다.

---

그림 2.1:  
VAE는 관측된 공간 $\mathbf{x}$와 잠재 공간 $\mathbf{z}$ 사이의 확률적 사상(stochastic mapping)을 학습한다.  
관측 공간의 경험적 분포 $q_D(\mathbf{x})$는 일반적으로 복잡하지만,  
잠재 공간의 분포는 비교적 단순할 수 있다 (예: 그림처럼 구형 분포).  

생성 모델은 $p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})$라는 결합 분포를 학습하며, 이는 보통 (항상은 아니지만) 다음과 같이 분해된다:

$$
p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) 
= p_{\boldsymbol{\theta}}(\mathbf{z}) \, p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})
$$

여기서 $p_{\boldsymbol{\theta}}(\mathbf{z})$는 잠재 공간에 대한 사전분포(prior distribution)이며,  
$p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$는 확률적 디코더(stochastic decoder)이다.  

확률적 인코더(stochastic encoder) $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$,  
즉 추론 모델(inference model)은 생성 모델의 실제 사후분포 $p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})$를 근사한다.  

<img src="/assets/img/books/intro-to-vae/2/image_1.png" alt="image" width="600px"> 

---

임의의 추론 모델 $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$, 즉, 변분 매개변수 $\boldsymbol{\phi}$의 어떤 선택에 대해서도 다음이 성립한다:

$$
\begin{align}
\log p_{\boldsymbol{\theta}}(\mathbf{x})
&= \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}
   \Big[ \log p_{\boldsymbol{\theta}}(\mathbf{x}) \Big]
   \tag{2.5} \\[6pt]
&= \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}
   \left[
     \log \!\left[
       \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}
            {p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})}
     \right]
   \right]
   \tag{2.6} \\[6pt]
&= \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}
   \left[
     \log \!\left[
       \frac{
         p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
         q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})
       }{
         q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})
         p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})
       }
     \right]
   \right]
   \tag{2.7} \\[6pt]
&=
\underbrace{
  \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}
  \left[
    \log \!\left[
      \frac{
        p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
      }{
        q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})
      }
    \right]
  \right]
}_{\displaystyle =\,\mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x}) \;(\text{ELBO})}
\;+\;
\underbrace{
  \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}
  \left[
    \log \!\left[
      \frac{
        q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})
      }{
        p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})
      }
    \right]
  \right]
}_{\displaystyle =\,D_{\mathrm{KL}}
  (q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})
  \,||\,
  p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x}))}
\tag{2.8}
\end{align}
$$

> 식 (2.5)는 확률변수의 기댓값 성질에 기반한다.  
>
> $\log p_{\boldsymbol{\theta}}(\mathbf{x})$는 $\mathbf{z}$ 와 무관한 상수항이므로,  
>  
> $$
> \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}
> [\,\log p_{\boldsymbol{\theta}}(\mathbf{x})\,]
> = \log p_{\boldsymbol{\theta}}(\mathbf{x})
> $$
>  
> 가 성립한다.  

> 식 (2.6)은 결합 확률의 정의에서 유도된다.  
>  
> 결합 분포 $p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})$는 사후분포와 주변분포의 곱으로 표현될 수 있다:
>  
> $$
> p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
> = p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x}) \, p_{\boldsymbol{\theta}}(\mathbf{x})
> $$
>  
> 이를 로그에 대입하면
>  
> $$
> \log p_{\boldsymbol{\theta}}(\mathbf{x})
> = \log \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}
>             {p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})}
> $$
>  
> 이 되고, 양변의 기댓값을 취함으로써 식 (2.6)이 성립한다.

---

식 (2.8)의 두 번째 항은 $ q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) $와 $ p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x}) $ 사이의 Kullback–Leibler (KL) 발산으로,  
항상 0 이상인 비음수(non-negative) 값을 가진다.

$$
D_{\mathrm{KL}}\!\left(q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \,\|\, p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})\right) \ge 0
\tag{2.9}
$$

그리고 두 분포가 정확히 일치할 때, 즉 $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) = p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})$일 때만 0이 된다.  

식 (2.8)의 첫 번째 항은 변분 하한(variational lower bound),  
즉 ELBO (Evidence Lower Bound)라고 부른다.

$$
\mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})
= \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}
  \big[ \log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
  - \log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \big]
\tag{2.10}
$$

KL 발산이 비음수이므로, ELBO는 데이터 로그 가능도의 하한(lower bound)이 된다.

$$
\begin{align}
\mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})
&= \log p_{\boldsymbol{\theta}}(\mathbf{x})
   - D_{\mathrm{KL}}\!\left(
      q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})
      \,\|\, 
      p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})
   \right)
   \tag{2.11} \\[6pt]
&\le \log p_{\boldsymbol{\theta}}(\mathbf{x})
   \tag{2.12}
\end{align}
$$

> (2.10) → (2.11) 유도 요약  
>
> 1) 베이즈 법칙 사용: $ p_{\theta}(x,z) = p_{\theta}(z\mid x)\,p_{\theta}(x) $   
> 2) 이를 (2.10)에 대입하고 항을 분리한다.
>
> $$
> \begin{aligned}
> \mathcal{L}_{\theta,\phi}(x)
> &= \mathbb{E}_{q_{\phi}(z\mid x)}
>    \big[ \log p_{\theta}(x,z) - \log q_{\phi}(z\mid x) \big] \\[4pt]
> &= \mathbb{E}_{q_{\phi}(z\mid x)}
>    \big[ \log p_{\theta}(z\mid x) + \log p_{\theta}(x) - \log q_{\phi}(z\mid x) \big] \\[4pt]
> &= \underbrace{\mathbb{E}_{q_{\phi}(z\mid x)}[\log p_{\theta}(x)]}_{=\;\log p_{\theta}(x)}
>    \;+\; \mathbb{E}_{q_{\phi}(z\mid x)}
>    \big[ \log p_{\theta}(z\mid x) - \log q_{\phi}(z\mid x) \big] \\[4pt]
> &= \log p_{\theta}(x)
>    - \mathbb{E}_{q_{\phi}(z\mid x)}
>      \big[ \log q_{\phi}(z\mid x) - \log p_{\theta}(z\mid x) \big] \\[4pt]
> &= \log p_{\theta}(x)
>    - D_{\mathrm{KL}}\!\big(q_{\phi}(z\mid x)\,\|\,p_{\theta}(z\mid x)\big),
> \end{aligned}
> $$
>
> 이로써 (2.11)이 성립한다.  
> 마지막 줄은 KL 발산의 정의 $ D_{\mathrm{KL}}(q\|p)=\mathbb{E}_q[\log q - \log p] $를 사용한 것이다.

따라서 KL 발산 $D_{\mathrm{KL}}(q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \| p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x}))$ 은 두 가지 “거리”를 결정한다.

1. 정의상, 근사 사후분포(approximate posterior)가  
   실제 사후분포(true posterior)와 얼마나 다른지를 나타내는 거리.  
2. ELBO $$\mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})$$ 와 로그 가능도 $\log p_{\boldsymbol{\theta}}(\mathbf{x})$ 사이의 차이(gap).  
   이를 '하한의 밀착 정도(tightness of the bound)'라고 한다.  
   $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$가 실제 사후분포 $p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})$를 더 잘 근사할수록  
   KL 발산이 작아지고, 두 값의 간극(gap)도 작아진다.

> KL 발산이 두 가지 “거리(distance)” 역할을 한다는 의미를 좀 더 자세히 살펴보면 다음과 같다.  
>
> **(1) 근사 사후분포와 실제 사후분포 간의 거리**  
>
> - $ D_{\mathrm{KL}}\\big(q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \,\|\, p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})\big) $ 은  
>   변분 추론(variational inference)에서 가장 핵심적인 오차 척도이다.  
> - 실제 사후분포 $ p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x}) $ 는 계산이 불가능(intractable)하므로,  
>   우리는 대신 근사 가능한 분포 $ q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) $ 를 학습시켜 이를 근사한다.  
> - KL 발산이 0이 되면, $ q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) = p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x}) $ 가 되어  
>   근사 사후분포가 완벽하게 실제 사후분포를 복원한다는 뜻이다.  
> - 반대로 KL 발산이 커질수록, 두 분포가 멀리 떨어져 있다는 의미이다.  
>
> **(2) ELBO와 로그 가능도(log-likelihood) 간의 차이(gap)**  
>
> - 식 (2.11)에서 보듯,  
>
   $$
   \mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})
   = \log p_{\boldsymbol{\theta}}(\mathbf{x})
     - D_{\mathrm{KL}}\!\big(q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})
     \,\|\, p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})\big)
   $$
>
>   이므로, KL 발산 항이 바로 로그 가능도와 ELBO 사이의 간극(gap)이 된다.  
> - 따라서 KL 발산은 '하한(lower bound)'이 얼마나 실제 로그 가능도에 가까운지를  
>   결정짓는 밀착 정도(tightness of the bound)를 나타낸다.  
> - 즉, $ q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) $ 가 실제 사후분포 $ p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x}) $를 잘 근사할수록 KL 발산이 작아지고,  
>   두 값 — ELBO와 로그 가능도 — 의 차이 또한 작아져 ELBO가 더 ‘타이트’해진다.  
>
> 요약하자면, KL 발산은  
> (a) 근사 추론의 품질을 평가하는 척도이자,  
> (b) ELBO가 실제 로그 가능도에 얼마나 근접한지를 알려주는 척도  
> 두 역할을 동시에 수행한다.

---

그림 2.2:  
변분 오토인코더(variational autoencoder)의 계산 흐름(computational flow)을 단순화하여 나타낸 개략도이다.  

<img src="/assets/img/books/intro-to-vae/2/image_2.png" alt="image" width="720px"> 

---

## 2.2.1 하나로 두 가지 효과 (Two for One)  

식 (2.11)을 살펴보면, 매개변수 $\boldsymbol{\theta}$ 와 $\boldsymbol{\phi}$ 에 대해 ELBO $\mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})$ 를 최대화하는 것은  
다음의 두 가지 목표를 동시에 최적화함을 알 수 있다.  

1. 주변우도 $p_{\boldsymbol{\theta}}(\mathbf{x})$ 를 근사적으로 최대화한다.  
   → 즉, 생성 모델(generative model)이 점점 더 나은 모델이 된다.  

2. 근사 사후분포 $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$와 실제 사후분포 $p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})$ 사이의 KL 발산을 최소화한다.  
   → 즉, 추론 모델(inference model) 역시 점점 더 정확해진다.  

---

## 2.3 확률적 경사 기반 ELBO 최적화  

ELBO의 중요한 성질 중 하나는, 모든 매개변수(즉, $\boldsymbol{\phi}$ 와 $\boldsymbol{\theta}$)에 대해  
확률적 경사하강법(stochastic gradient descent, SGD)으로  
공동 최적화(joint optimization)가 가능하다는 점이다.  

임의의 초기값 $\boldsymbol{\phi}$ 와 $\boldsymbol{\theta}$로 시작하여, 수렴할 때까지 확률적으로 그 값을 반복적으로 최적화할 수 있다.  

독립이고 동일하게 분포된(i.i.d.) 데이터셋 $\mathcal{D}$ 가 주어졌을 때,  
ELBO 목적함수는 각 데이터 포인트별 ELBO의 합(또는 평균)으로 표현된다.  

$$
\mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathcal{D})
= \sum_{\mathbf{x} \in \mathcal{D}}
  \mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})
\tag{2.13}
$$  

각 데이터 포인트에 대한 ELBO와 그 그래디언트 $\nabla_{\boldsymbol{\theta}, \boldsymbol{\phi}} \mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})$ 는 일반적으로 계산이 불가능(intractable)하다.  

그러나 뒤에서 보게 될 것처럼,  
좋은 불편 추정량(unbiased estimator)인 $$\tilde{\nabla}_{\boldsymbol{\theta}, \boldsymbol{\phi}} \mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})$$를 사용하면,  
미니배치 SGD도 수행할 수 있다.  

---

불편향(unbiased) ELBO 그래디언트는  
생성 모델의 매개변수 $\boldsymbol{\theta}$에 대해 비교적 간단하게 계산할 수 있다.

$$
\begin{align}
\nabla_{\boldsymbol{\theta}}
  \mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})
&= \nabla_{\boldsymbol{\theta}}
   \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}
   \big[
     \log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
     - \log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})
   \big]
   \tag{2.14} \\[6pt]
&= \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}
   \big[
     \nabla_{\boldsymbol{\theta}}
     \big(
       \log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
       - \log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})
     \big)
   \big]
   \tag{2.15} \\[6pt]
&\simeq
   \nabla_{\boldsymbol{\theta}}
   \big(
     \log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
     - \log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})
   \big)
   \tag{2.16} \\[6pt]
&=
   \nabla_{\boldsymbol{\theta}}
   \log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
   \tag{2.17}
\end{align}
$$

마지막 줄의 식 (2.17)은 식 (2.15)의 단순한 몬테카를로 추정(Monte Carlo estimator)이며,  
식 (2.16)과 (2.17)에 등장하는 $\mathbf{z}$는 $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$ 로부터 샘플링된 확률표본이다.  

> **1) (2.14) → (2.15):**  
>    그래디언트 $\nabla_{\theta}$는 디코더 $p_{\theta}(x,z)$ 의 파라미터에만 의존하므로,  
>    인코더 분포 $q_{\phi}(z\mid x)$와는 무관하다.  
>    따라서 그래디언트 $\nabla_{\theta}$를 $q_{\phi}(z\mid x)$에 대한 기댓값 연산자 안쪽으로 이동할 수 있다.  
>
>    $$
>    \nabla_{\theta}\,
>    \mathbb{E}_{q_{\phi}(z\mid x)}[f(z)]
>    =
>    \mathbb{E}_{q_{\phi}(z\mid x)}[\nabla_{\theta} f(z)]
>    $$
>
>    이를 적용하면 다음과 같이 쓸 수 있다.
>
>    $$
>    \nabla_{\theta}
>    \mathcal{L}_{\theta,\phi}(x)
>    =
>    \mathbb{E}_{q_{\phi}(z\mid x)}
>    \big[
>      \nabla_{\theta}
>      (\log p_{\theta}(x,z)
>      - \log q_{\phi}(z\mid x))
>    \big]
>    \tag{2.15}
>    $$
>
> ---
>
> **2) (2.15) → (2.16):**  
>    (2.15)식의 기댓값은 적분 형태로 계산하기 어렵기 때문에,  
>    몬테카를로 추정(Monte Carlo estimation)을 이용해 근사한다.  
>
>    $$
>    \mathbb{E}_{q_{\phi}(z\mid x)}[f(z)]
>    \approx
>    \frac{1}{L}\sum_{l=1}^{L} f(z^{(l)}),
>    \quad z^{(l)} \sim q_{\phi}(z\mid x)
>    $$
>
>    샘플 수 $L$이 충분히 크면 이 근사치는 실제 기댓값에 수렴하며,  
>    이는 대수의 법칙(Law of Large Numbers)에 의해 보장된다.  
>
>    이를 (2.15)에 적용하면,
>
>    $$
>    \mathbb{E}_{q_{\phi}(z\mid x)}
>    [\nabla_{\theta}(\log p_{\theta}(x,z) - \log q_{\phi}(z\mid x))]
>    \;\simeq\;
>    \nabla_{\theta}
>    (\log p_{\theta}(x,z) - \log q_{\phi}(z\mid x))
>    \tag{2.16}
>    $$
>
>    즉, $q_{\phi}(z\mid x)$ 로부터 하나 또는 여러 개의 샘플을 뽑아, 그 평균으로 기댓값을 근사한다.  
>
> ---
>
> **3) (2.16) → (2.17):**  
>    $\log q_{\phi}(z\mid x)$ 는 인코더 파라미터 $\phi$ 의 함수이므로 $\theta$ 에 대한 미분은 0이다.  
>    따라서 남는 항은 $\log p_{\theta}(x,z)$에 대한 그래디언트뿐이다.
>
>    $$
>    \nabla_{\theta}\mathcal{L}_{\theta,\phi}(x)
>    \simeq
>    \nabla_{\theta}\log p_{\theta}(x,z)
>    \tag{2.17}
>    $$

---

불편향(unbiased) 그래디언트를 변분 파라미터(variational parameters) $\boldsymbol{\phi}$ 에 대해 계산하는 것은  
$\boldsymbol{\theta}$ 에 대한 경우보다 훨씬 더 어렵다.  

이는 ELBO의 기댓값이 분포 $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ 에 대해 정의되어 있는데,  
이 분포 자체가 $\boldsymbol{\phi}$ 의 함수이기 때문이다.  

즉, 일반적으로 다음이 성립한다.

$$
\begin{align}
\nabla_{\boldsymbol{\phi}}
  \mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})
&= \nabla_{\boldsymbol{\phi}}
   \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}
   \big[
     \log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
     - \log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})
   \big]
   \tag{2.18} \\[6pt]
&\neq
   \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}
   \big[
     \nabla_{\boldsymbol{\phi}}
     \big(
       \log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
       - \log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})
     \big)
   \big]
   \tag{2.19}
\end{align}
$$

> 1) **각 파라미터의 역할**  
>    - $\boldsymbol{\theta}$ : 디코더(Decoder)의 파라미터로, 생성 모델 $p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})$를 구성한다.  
>      즉, 잠재변수 $\mathbf{z}$ 로부터 데이터를 복원(reconstruct)하는 역할을 한다.  
>    - $\boldsymbol{\phi}$ : 인코더(Encoder)의 파라미터로, 근사 사후분포 $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ 를 구성한다.  
>      즉, 입력 $\mathbf{x}$ 로부터 잠재변수 $\mathbf{z}$ 의 분포를 추정하는 역할을 한다.  
>
> ---
>
> 2) **디코더(θ)에 대한 기댓값 식**  
>    디코더의 경우 ELBO의 기댓값은 다음과 같이 주어진다.
>
>    $$
>    \mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})
>    = \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}
>      \big[\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
>      - \log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})\big]
>    $$
>
>    여기서 기댓값의 분포는 $\boldsymbol{\phi}$이고, 미분 연산자는 $\boldsymbol{\theta}$에 대해 작동한다.  
>
>    따라서 $\boldsymbol{\theta}$ 는 분포 $q_{\boldsymbol{\phi}}$ 와 독립적이므로, 미분을 기댓값 안쪽으로 자유롭게 옮길 수 있다.
>
>    $$
>    \nabla_{\boldsymbol{\theta}}
>    \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}[f(\mathbf{z})]
>    = \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}
>      [\nabla_{\boldsymbol{\theta}} f(\mathbf{z})]
>    $$
>
>    즉, “기댓값의 분포는 φ, 미분 대상은 θ”이므로 두 연산이 서로 간섭하지 않아 계산이 단순하다.  
>
> ---
>
> 3) **인코더(φ)에 대한 기댓값 식**  
>    반면 인코더의 경우 ELBO의 기댓값은 다음과 같다.
>
>    $$
>    \mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})
>    = \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}
>      \big[\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
>      - \log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})\big]
>    $$
>
>    이번에는 기댓값의 분포도 φ에 의존하고, 기댓값 안의 항(특히 $\log q_{\boldsymbol{\phi}}$) 역시 φ에 의존한다.  
>
>    즉, 기댓값을 적분 형태로 쓰면 다음과 같다.
>
>    $$
>    \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}[f(\mathbf{z})]
>    = \int f(\mathbf{z})\, q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})\, d\mathbf{z}
>    $$
>
>    이때 $\boldsymbol{\phi}$ 에 대해 미분하면,
>
>    $$
>    \nabla_{\boldsymbol{\phi}}
>    \int f(\mathbf{z})\, q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})\, d\mathbf{z}
>    =
>    \int \nabla_{\boldsymbol{\phi}} f(\mathbf{z})\, q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})\, d\mathbf{z}
>    + \int f(\mathbf{z})\, \nabla_{\boldsymbol{\phi}} q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})\, d\mathbf{z}
>    $$
>
>    두 번째 항(분포 자체의 미분)이 추가되므로, 단순히 기댓값 안쪽만 미분한 것과는 달라진다.  
>
>    즉,
>
>    $$
>    \nabla_{\boldsymbol{\phi}}
>    \mathbb{E}_{q_{\boldsymbol{\phi}}}[f(\mathbf{z})]
>    \neq
>    \mathbb{E}_{q_{\boldsymbol{\phi}}}[\nabla_{\boldsymbol{\phi}} f(\mathbf{z})]
>    $$
>
>    이는 기댓값의 분포와 미분 대상이 모두 φ에 의존하기 때문이다.  
>
>    따라서 인코더의 경우에는  
>    “기댓값 안의 식”과 “확률밀도 함수” 모두에 $\boldsymbol{\phi}$ 가 들어 있어 미분이 훨씬 더 복잡해진다.  

연속적인 잠재변수의 경우, $\nabla_{\boldsymbol{\theta}, \boldsymbol{\phi}} \mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})$의 불편향 추정값(unbiased estimates)을 계산하기 위해  
재매개변수화 기법(reparameterization trick)을 사용할 수 있다.  

이 확률적 추정은 확률적 경사하강법(SGD)을 사용하여 ELBO를 최적화할 수 있게 해 준다.  
알고리즘 1을 참조하라.  

불연속 잠재변수에 대한 변분 방법(variational methods)에 대한 논의는 2.9.1절을 참조하라.

---

**알고리즘 1. ELBO의 확률적 최적화 (Stochastic optimization of the ELBO)**

노이즈는 미니배치 샘플링과 $p(\boldsymbol{\epsilon})$의 샘플링 두 과정 모두에서 발생하기 때문에,  
이 절차는 이중 확률적 최적화(doubly stochastic optimization) 방식이다.  

이 절차는  
오토인코딩 변분 베이즈(Auto-Encoding Variational Bayes, AEVB) 알고리즘이라고도 불린다.

**데이터 (Data):**  
- $\mathcal{D}$: 데이터셋 (Dataset)  
- $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$: 추론 모델 (Inference model)  
- $p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})$: 생성 모델 (Generative model)  

**결과 (Result):**  
- $\boldsymbol{\theta}, \boldsymbol{\phi}$: 학습된 파라미터 (Learned parameters)  

**절차 (Procedure):**

$$
\begin{aligned}
(\boldsymbol{\theta}, \boldsymbol{\phi}) &\leftarrow \text{Initialize parameters} \\[4pt]
\textbf{while } &\text{SGD not converged do} \\[2pt]
&\mathcal{M} \sim \mathcal{D} 
\quad \text{(Random minibatch of data)} \\[4pt]
&\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon}) 
\quad \text{(Random noise for every datapoint in } \mathcal{M}\text{)} \\[4pt]
&\text{Compute } 
\tilde{\mathcal{L}}_{\boldsymbol{\theta},\boldsymbol{\phi}}(\mathcal{M}, \boldsymbol{\epsilon})
\text{ and its gradients }
\nabla_{\boldsymbol{\theta},\boldsymbol{\phi}}
\tilde{\mathcal{L}}_{\boldsymbol{\theta},\boldsymbol{\phi}}(\mathcal{M}, \boldsymbol{\epsilon}) \\[4pt]
&\text{Update } \boldsymbol{\theta} \text{ and } \boldsymbol{\phi}
\text{ using SGD optimizer} \\[4pt]
\textbf{end}
\end{aligned}
$$

> 1) **초기화 단계**  
>    $(\boldsymbol{\theta}, \boldsymbol{\phi}) \leftarrow$ Initialize parameters  
>    - 생성 모델(디코더)의 파라미터 $\boldsymbol{\theta}$ 와  
>      추론 모델(인코더)의 파라미터 $\boldsymbol{\phi}$ 를 무작위로 초기화한다.  
>
> ---
>
> 2) **학습 반복 루프 (while SGD not converged)**  
>    확률적 경사하강법(SGD)을 이용하여 ELBO를 최적화할 때까지  
>    다음 과정을 반복한다.  
>
>    - $\mathcal{M} \sim \mathcal{D}$  
>      → 전체 데이터셋 $\mathcal{D}$ 에서 무작위 미니배치(minibatch) $\mathcal{M}$ 을 샘플링한다.  
>
>    - $\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})$  
>      → 각 데이터 포인트에 대해 독립적인 노이즈 $\boldsymbol{\epsilon}$ 을 샘플링한다.  
>         (예: $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$)  
>         이는 재매개변수화 기법에서 사용되는 확률 변수이다.  
>
>    - $$\tilde{\mathcal{L}}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathcal{M}, \boldsymbol{\epsilon})$$ 계산  
>      → 주어진 미니배치와 노이즈를 이용해 ELBO의 근사값을 계산한다.  
>         이때 $\mathbf{z} = g_{\boldsymbol{\phi}}(\mathbf{x}, \boldsymbol{\epsilon})$ 형태로 잠재변수를 샘플링하여 기대값을 근사한다.  
>
>    - $\nabla_{\boldsymbol{\theta}, \boldsymbol{\phi}} \tilde{\mathcal{L}}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathcal{M}, \boldsymbol{\epsilon})$ 계산  
>      → 계산된 ELBO에 대해 $\boldsymbol{\theta}$ 와 $\boldsymbol{\phi}$ 각각의 그래디언트를 구한다.  
>
>    - 파라미터 업데이트  
>      → 계산된 그래디언트를 이용하여 SGD 옵티마이저로 $\boldsymbol{\theta}$ 와 $\boldsymbol{\phi}$ 를 갱신한다.  
>
> ---
>
> 3) **종료 조건 (end)**  
>    - SGD가 수렴(converged)하면 학습을 종료한다.  
>    - 최종적으로 학습된 파라미터 $(\boldsymbol{\theta}, \boldsymbol{\phi})$ 는  
>      디코더와 인코더의 최적화된 형태가 된다.

---

## 2.4 재매개변수화 기법 (Reparameterization Trick)

연속적인 잠재변수와 미분 가능한 인코더 및 생성 모델의 경우,  
ELBO는 변수 변경(change of variables)을 통해 $\boldsymbol{\phi}$ 와 $\boldsymbol{\theta}$ 모두에 대해 직접적으로 미분할 수 있다.  

이 방법을 재매개변수화 기법(Reparameterization trick)이라고 하며,  
Kingma & Welling (2014), Rezende et al. (2014) 에 의해 제안되었다.

---

### 2.4.1 변수 변경 (Change of Variables)

먼저, 잠재변수 $\mathbf{z}$ 가 인코더 분포 $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$로부터 샘플링된다고 하자.  

이때 $\mathbf{z}$를 또 다른 확률변수 $\boldsymbol{\epsilon}$의  
미분 가능하고 가역적인(invertible) 변환(transformation)으로 표현할 수 있다.

$$
\mathbf{z} = g(\boldsymbol{\epsilon}, \boldsymbol{\phi}, \mathbf{x})
\tag{2.20}
$$

여기서 $\boldsymbol{\epsilon}$은 $\mathbf{x}$ 나 $\boldsymbol{\phi}$와는 독립적인 확률변수이다.

---

### 2.4.2 변수 변경 하에서의 기댓값의 그래디언트

이와 같은 변수 변경이 되면, 기댓값은 $\boldsymbol{\epsilon}$에 대한 형태로 다시 쓸 수 있다.

$$
\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}[f(\mathbf{z})]
= \mathbb{E}_{p(\boldsymbol{\epsilon})}[f(\mathbf{z})]
\tag{2.21}
$$

여기서 $\mathbf{z} = g(\boldsymbol{\epsilon}, \boldsymbol{\phi}, \mathbf{x})$ 이다.  

이 경우 기댓값 연산자와 그래디언트 연산자는 교환 가능하게 되며,  
간단한 몬테카를로 추정량(Monte Carlo estimator)을 구성할 수 있다.

$$
\begin{align}
\nabla_{\boldsymbol{\phi}}
  \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}[f(\mathbf{z})]
&= \nabla_{\boldsymbol{\phi}}
   \mathbb{E}_{p(\boldsymbol{\epsilon})}[f(\mathbf{z})]
   \tag{2.22} \\[6pt]
&= \mathbb{E}_{p(\boldsymbol{\epsilon})}
   [\nabla_{\boldsymbol{\phi}} f(\mathbf{z})]
   \tag{2.23} \\[6pt]
&\simeq \nabla_{\boldsymbol{\phi}} f(\mathbf{z})
   \tag{2.24}
\end{align}
$$

마지막 식에서는, 무작위 노이즈 표본 $\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})$ 에 대해
$\mathbf{z} = g(\boldsymbol{\phi}, \mathbf{x}, \boldsymbol{\epsilon})$ 이다.  

> 1) 추정량(estimator) 이란  
>    실제 기댓값이나 적분을 계산하기 어려울 때,  
>    샘플 데이터를 이용하여 그 값을 근사적으로 계산하는 통계적 도구를 말한다.  
>
> 2) 몬테카를로 추정량(Monte Carlo estimator)은  
>    확률분포에서 여러 개의 샘플을 뽑아,  
>    그 결과들의 평균으로 기댓값을 근사하는 방법이다.  
>
>    예를 들어,  
>
>    $$
>    \mathbb{E}_{p(\boldsymbol{\epsilon})}[f(\boldsymbol{\epsilon})]
>    \approx
>    \frac{1}{L}\sum_{l=1}^{L} f(\boldsymbol{\epsilon}^{(l)}),
>    \quad \boldsymbol{\epsilon}^{(l)} \sim p(\boldsymbol{\epsilon})
>    $$
>
>    와 같이 표현할 수 있다.  
>
> 3) 위 식에서 $L$은 샘플의 개수이며,  
>    $L$이 커질수록 근사값은 실제 기댓값에 가까워진다.  
>    이는 대수의 법칙(Law of Large Numbers)에 의해 보장된다.  
>
> 4) 따라서 (2.22)~(2.24)에서처럼  
>    기댓값과 그래디언트가 교환 가능한 경우,  
>    실제 적분을 계산하지 않고도  
>    확률적 샘플을 이용해 그래디언트를 근사할 수 있다.  
>
>    이러한 방식으로 얻은 추정값이  
>    바로 몬테카를로 추정량(Monte Carlo estimator)이다.

그림 2.3은 이에 대한 시각적 예시 및 추가 설명을 제공하며,  
그림 3.2는 2차원 예제(2D toy problem)에 대해  
결과로 얻어지는 사후분포(posteriors)를 시각화한 것이다.

---

그림 2.3: 재매개변수화 기법(Reparameterization trick)의 예시  

변분 파라미터 $\boldsymbol{\phi}$는 확률변수 $\mathbf{z} \sim q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$를 통해 목적함수 $f$ 에 영향을 미친다.  

우리는 확률적 경사하강법(SGD)을 이용해 목적함수를 최적화하기 위해 $\nabla_{\boldsymbol{\phi}} f$ 를 계산하고자 한다.  

원래 형태(왼쪽)에서는,  
확률 변수 $\mathbf{z}$ 를 통해 그래디언트를 직접 역전파(backpropagate)할 수 없기 때문에  
$f$ 를 $\boldsymbol{\phi}$ 에 대해 미분할 수 없다.

우리는 변수 $\mathbf{z}$의 무작위성을,  
$\boldsymbol{\phi}$, $\mathbf{x}$, 그리고 새로 도입된 확률 변수 $\boldsymbol{\epsilon}$의 결정론적이고 미분 가능한 함수로  
변수의 매개변수를 다시 설정(re-parameterizing)함으로써  
‘외부화(externalize)’할 수 있다.

이것은 우리가 ‘$\mathbf{z}$를 통해 역전파(backprop through $\mathbf{z}$)’하고,  
그래디언트 $\nabla_{\boldsymbol{\phi}} f$ 를 계산할 수 있게 해준다.

<img src="/assets/img/books/intro-to-vae/2/image_3.png" alt="image" width="720px"> 

---

### 2.4.3 ELBO의 그래디언트

재매개변수화(reparameterization) 하에서는,  
기댓값을 $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ 에 대한 것에서 $p(\boldsymbol{\epsilon})$ 에 대한 것으로 대체할 수 있다.  
이때 ELBO는 다음과 같이 다시 쓸 수 있다.

$$
\begin{align}
\mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})
&= \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}
   [\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
   - \log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})]
   \tag{2.25} \\[6pt]
&= \mathbb{E}_{p(\boldsymbol{\epsilon})}
   [\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
   - \log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})]
   \tag{2.26}
\end{align}
$$

여기서 $\mathbf{z} = g(\boldsymbol{\epsilon}, \boldsymbol{\phi}, \mathbf{x})$ 이다.

---

그 결과, 단일 데이터 포인트에 대한  
ELBO의 단순한 몬테카를로 추정량(estimator) $\tilde{\mathcal{L}}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})$ 을 구성할 수 있다.  

이때 하나의 노이즈 샘플 $\boldsymbol{\epsilon}$을 $p(\boldsymbol{\epsilon})$ 로부터 사용한다.

$$
\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})
\tag{2.27}
$$

$$
\mathbf{z} = g(\boldsymbol{\phi}, \mathbf{x}, \boldsymbol{\epsilon})
\tag{2.28}
$$

$$
\tilde{\mathcal{L}}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})
= \log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
  - \log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})
\tag{2.29}
$$

---

이 일련의 연산들은 TensorFlow와 같은 소프트웨어에서  
기호적 그래프(symbolic graph) 형태로 표현될 수 있으며,  
파라미터 $\boldsymbol{\theta}$ 와 $\boldsymbol{\phi}$에 대해 손쉽게 미분될 수 있다.  

그 결과로 얻어진 그래디언트 $\nabla_{\boldsymbol{\phi}} \tilde{\mathcal{L}}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})$는   
미니배치 SGD를 사용하여 ELBO를 최적화하는 데 사용된다.  
알고리즘 1을 참조하라.  

이 알고리즘은 Kingma와 Welling(2014)에 의해 처음에는  
오토인코딩 변분 베이즈(Auto-Encoding Variational Bayes, AEVB) 알고리즘으로 불렸다.  

보다 일반적으로, 재매개변수화된 ELBO 추정량은  
확률적 그래디언트 변분 베이즈(Stochastic Gradient Variational Bayes, SGVB)  
추정량이라고 불린다.  

이 추정량은 또한 모델 파라미터에 대한 사후분포(posterior)를 추정하는 데에도 사용할 수 있으며,  
이에 대한 설명은 Kingma와 Welling(2014)의 부록(Appendix)에 제시되어 있다.

---

#### 불편향성 (Unbiasedness)

이 그래디언트는  
정확한 단일 데이터 포인트 ELBO 그래디언트의 불편향 추정량(unbiased estimator)이다.  

즉, 노이즈 $\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})$에 대해 평균을 취하면,  
이 그래디언트는 단일 데이터 포인트 ELBO 그래디언트와 동일하다.

$$
\begin{align}
\mathbb{E}_{p(\boldsymbol{\epsilon})}
\big[
\nabla_{\boldsymbol{\theta}, \boldsymbol{\phi}}
\tilde{\mathcal{L}}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x}; \boldsymbol{\epsilon})
\big]
&=
\mathbb{E}_{p(\boldsymbol{\epsilon})}
\big[
\nabla_{\boldsymbol{\theta}, \boldsymbol{\phi}}
(\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
- \log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x}))
\big]
\tag{2.30} \\[8pt]
&=
\nabla_{\boldsymbol{\theta}, \boldsymbol{\phi}}
\big(
\mathbb{E}_{p(\boldsymbol{\epsilon})}
[\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
- \log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})]
\big)
\tag{2.31} \\[8pt]
&=
\nabla_{\boldsymbol{\theta}, \boldsymbol{\phi}}
\mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})
\tag{2.32}
\end{align}
$$

---

### 2.4.4 $\log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$의 계산

ELBO (또는 그 추정량)의 계산은,  
$\mathbf{x}$ 의 값이 주어지고, $\mathbf{z}$ 또는 동등하게 $\boldsymbol{\epsilon}$의 값이 주어졌을 때,  
밀도 $\log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$의 계산을 필요로 한다.

이 로그 밀도의 계산은, 적절한 변환 함수 $g()$ 를 선택하기만 한다면, 비교적 간단한 연산이다.

---

일반적으로 우리는 밀도 $p(\boldsymbol{\epsilon})$를 알고 있다. 이는 선택된 노이즈 분포의 밀도이기 때문이다.  

함수 $g(\cdot)$ 가 가역(invertible) 함수라면, $\boldsymbol{\epsilon}$ 과 $\mathbf{z}$ 의 밀도는 다음과 같은 관계를 가진다.

$$
\log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})
= \log p(\boldsymbol{\epsilon})
- \log d_{\boldsymbol{\phi}}(\mathbf{x}, \boldsymbol{\epsilon})
\tag{2.33}
$$

여기서 두 번째 항은 야코비안 행렬(Jacobian matrix)  
$\dfrac{\partial \mathbf{z}}{\partial \boldsymbol{\epsilon}}$ 의 행렬식(determinant)의 절댓값에 대한 로그이다.

$$
\log d_{\boldsymbol{\phi}}(\mathbf{x}, \boldsymbol{\epsilon})
= \log \left| \det\!\left(\frac{\partial \mathbf{z}}{\partial \boldsymbol{\epsilon}}\right) \right|
\tag{2.34}
$$

우리는 이것을 $\boldsymbol{\epsilon}$ 에서 $\mathbf{z}$ 로의 변환의 로그-행렬식(log-determinant)이라고 부른다.  

> 식 (2.33)은 확률변수의 변수변환 공식(change of variables formula)에서 나온다.  
>  
> 만약 $\mathbf{z}$가 $\mathbf{z} = g_{\boldsymbol{\phi}}(\mathbf{x}, \boldsymbol{\epsilon})$로 정의되고, $g$가 가역(invertible)이라면,  
> 두 변수 $\mathbf{z}$와 $\boldsymbol{\epsilon}$의 확률밀도는 다음 관계를 가진다:
>
> $$
> q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})
> = p(\boldsymbol{\epsilon})
> \left|\det\!\left(\frac{\partial \boldsymbol{\epsilon}}{\partial \mathbf{z}}\right)\right|
> = p(\boldsymbol{\epsilon})
> \Big/\!
> \left|\det\!\left(\frac{\partial \mathbf{z}}{\partial \boldsymbol{\epsilon}}\right)\right|.
> $$
>  
> 양변에 로그를 취하면,
>
> $$
> \log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})
> = \log p(\boldsymbol{\epsilon})
> - \log \left|\det\!\left(\frac{\partial \mathbf{z}}{\partial \boldsymbol{\epsilon}}\right)\right|.
> $$
>  
> 이것이 바로 식 (2.33)의 형태이다.  
>  
> 즉, $\boldsymbol{\epsilon}$의 분포를 알고 있을 때, $\mathbf{z}$로의 변환이 가역적이면  
> 밀도변환 공식에 의해 두 확률밀도 간의 관계를 로그-야코비안 항을 통해 간단히 연결할 수 있다.  
> 이 원리를 이용하면 VAE에서 샘플링 과정을 미분 가능한 형태로 표현할 수 있게 된다  
> (즉, reparameterization trick의 핵심 아이디어).

이 로그-행렬식이 $g(\cdot)$ 와 마찬가지로 $\mathbf{x}$, $\boldsymbol{\epsilon}$, 그리고 $\boldsymbol{\phi}$ 의 함수임을 명확히 하기 위해  
$\log d_{\boldsymbol{\phi}}(\mathbf{x}, \boldsymbol{\epsilon})$ 라는 표기를 사용한다.  

야코비안 행렬(Jacobian matrix)은 $\boldsymbol{\epsilon}$ 에서 $\mathbf{z}$ 로의 변환에 대한 모든 1차 미분항을 포함한다.

$$
\frac{\partial \mathbf{z}}{\partial \boldsymbol{\epsilon}}
= 
\frac{\partial (z_1, \dots, z_k)}{\partial (\epsilon_1, \dots, \epsilon_k)}
=
\begin{pmatrix}
\dfrac{\partial z_1}{\partial \epsilon_1} & \cdots & \dfrac{\partial z_1}{\partial \epsilon_k} \\[6pt]
\vdots & \ddots & \vdots \\[4pt]
\dfrac{\partial z_k}{\partial \epsilon_1} & \cdots & \dfrac{\partial z_k}{\partial \epsilon_k}
\end{pmatrix}
\tag{2.35}
$$

앞으로 보이겠지만,  
$\log d_{\boldsymbol{\phi}}(\mathbf{x}, \boldsymbol{\epsilon})$ 를 쉽게 계산할 수 있는 매우 유연한 변환 함수 $g(\cdot)$ 를 구성할 수 있다.  
이를 통해 매우 유연한 추론 모델 $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ 를 얻을 수 있다.

---

## 2.5 분리된 가우시안 사후분포 (Factorized Gaussian posteriors)

일반적인 선택은 단순한 분리된 가우시안 인코더(factorized Gaussian encoder)이다.

$$
q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})
= \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}, \mathrm{diag}(\boldsymbol{\sigma}^2))
\tag{2.36}
$$

$$
(\boldsymbol{\mu}, \log \boldsymbol{\sigma})
= \mathrm{EncoderNeuralNet}_{\boldsymbol{\phi}}(\mathbf{x})
\tag{2.36}
$$

$$
q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})
= \prod_i q_{\boldsymbol{\phi}}(z_i\mid\mathbf{x})
= \prod_i \mathcal{N}(z_i; \mu_i, \sigma_i^2)
\tag{2.37}
$$

여기서 $\mathcal{N}(z_i; \mu_i, \sigma_i^2)$는  
단변량 가우시안 분포(univariate Gaussian distribution)의 확률밀도함수(PDF)이다.  

> (1) 인코더 신경망의 역할  
>     인코더 $\mathrm{EncoderNeuralNet}_{\boldsymbol{\phi}}(\mathbf{x})$ 는 입력 $\mathbf{x}$ 로부터  
>     잠재분포의 파라미터인 평균 $\boldsymbol{\mu}$ 와 로그표준편차 $\log \boldsymbol{\sigma}$ 를 출력한다.  
>     즉, 인코더는 데이터 $\mathbf{x}$ 가 주어졌을 때 잠재변수 $\mathbf{z}$ 의 분포를 결정하는 함수이다.  
>
> (2) 인코더가 정의하는 분포의 형태  
>     인코더가 출력한 $\boldsymbol{\mu}$ 와 $\boldsymbol{\sigma}$ 는  
>     곧 잠재변수 $\mathbf{z}$ 가 따르는 분포의 파라미터로 사용된다.  
>     이때의 분포가 바로  
>     $$
>     q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})
>     = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}, \mathrm{diag}(\boldsymbol{\sigma}^2))
>     $$
>     이다.  
>     즉, 인코더는 입력 $\mathbf{x}$ 에 대해  
>     잠재공간(latent space)에서 하나의 다변량 정규분포를 정의하는 셈이다.  
>
> (3) 왜 “분리된(factorized)” 형태인가  
>     $\mathrm{diag}(\boldsymbol{\sigma}^2)$ 는 각 차원의 분산을 대각원소로 가지는 대각행렬이므로,  
>     서로 다른 잠재변수 $z_i$ 와 $z_j$ 사이의 공분산이 0이다.  
>     즉, 모든 잠재차원은 독립(independent)이다.  
>     그래서 이 분포를 분리된 가우시안 분포(factorized Gaussian)라고 부른다.  
>
> (4) 다변량 정규분포의 의미  
>     - $\mathbf{z}$ : 잠재변수 벡터  
>     - $\boldsymbol{\mu}$ : 각 잠재변수의 평균(mean)  
>     - $\mathrm{diag}(\boldsymbol{\sigma}^2)$ : 공분산 행렬(covariance matrix)  
>     즉, $\mathbf{z}$ 는 평균 $\boldsymbol{\mu}$ 를 중심으로  
>     각 차원별 표준편차 $\boldsymbol{\sigma}$ 를 가지는 독립 정규분포의 집합으로 구성된다.  
>
> (5) 결과적으로  
>     인코더는 확률분포 자체를 출력하지 않고,  
>     그 분포를 정의하는 파라미터 $(\boldsymbol{\mu}, \boldsymbol{\sigma})$ 를 예측한다.  
>     이후 이 두 값을 이용해  
>     잠재변수 $\mathbf{z}$ 를 샘플링하거나 ELBO 계산에 활용한다.

재매개변수화(reparameterization) 이후, 다음과 같이 쓸 수 있다:

$$
\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)
\tag{2.38}
$$

$$
(\boldsymbol{\mu}, \log \boldsymbol{\sigma})
= \mathrm{EncoderNeuralNet}_{\boldsymbol{\phi}}(\mathbf{x})
\tag{2.39}
$$

$$
\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}
\tag{2.40}
$$

여기서 $\odot$ 는 원소별(element-wise) 곱을 의미한다.  

> (1) 식 (2.38)의 의미  
>     $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$ 는  
>     평균이 0이고 분산이 1인 표준 정규분포로부터  
>     노이즈 벡터 $\boldsymbol{\epsilon}$ 을 샘플링한다는 뜻이다.  
>     이 노이즈는 학습 가능한 파라미터와는 무관하며,  
>     모델의 확률적 성질을 유지하기 위한 무작위성의 근원이다.  
>
> (2) 식 (2.39)의 의미  
>     인코더 신경망 $\mathrm{EncoderNeuralNet}_{\boldsymbol{\phi}}(\mathbf{x})$ 는  
>     입력 $\mathbf{x}$ 로부터 잠재변수 분포의 평균 $\boldsymbol{\mu}$ 와 로그표준편차 $\log \boldsymbol{\sigma}$ 를 예측한다.  
>     즉, 인코더는 $\mathbf{x}$ 가 주어졌을 때 $\mathbf{z}$ 의 분포를 정의한다.  
>
> (3) 식 (2.40)의 의미  
>     $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ 은  
>     노이즈 $\boldsymbol{\epsilon}$ 을 결정적 함수로 변환하여 샘플링을 표현한 것이다.  
>     여기서 $\odot$ 는 원소별 곱(element-wise product)을 나타낸다.  
>     이 방식 덕분에 $\boldsymbol{\mu}$ 와 $\boldsymbol{\sigma}$ 에 대해  
>     미분이 가능해지며, 역전파(backpropagation)가 가능하다.  
>
> (4) 요약  
>     즉, 확률적 샘플링을 “결정적 함수 + 외부 노이즈” 형태로 바꾼 것이  
>     재매개변수화 트릭(reparameterization trick)이다.  
>     이를 통해 기대값 연산 안에서도 그래디언트를 전파할 수 있다.

$\boldsymbol{\epsilon}$ 에서 $\mathbf{z}$ 로의 변환에 대한 야코비안(Jacobian)은 다음과 같다:

$$
\frac{\partial \mathbf{z}}{\partial \boldsymbol{\epsilon}} = \mathrm{diag}(\boldsymbol{\sigma})
\tag{2.41}
$$

즉, $\boldsymbol{\sigma}$ 의 원소들이 대각선에 놓인 대각행렬(diagonal matrix)이다.  
대각행렬(또는 보다 일반적으로 삼각행렬)의 행렬식(determinant)은 대각 원소들의 곱이다.  
따라서 야코비안의 로그 행렬식(log determinant)은 다음과 같다.

$$
\log d_{\boldsymbol{\phi}}(\mathbf{x}, \boldsymbol{\epsilon})
= \log \left|\det\!\left(\frac{\partial \mathbf{z}}{\partial \boldsymbol{\epsilon}}\right)\right|
= \sum_i \log \sigma_i
\tag{2.42}
$$

따라서 사후 확률밀도(posteriordensity)는 다음과 같다.

$$
\log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})
= \log p(\boldsymbol{\epsilon})
- \log d_{\boldsymbol{\phi}}(\mathbf{x}, \boldsymbol{\epsilon})
\tag{2.43}
$$

$$
= \sum_i (\log \mathcal{N}(\epsilon_i; 0, 1)
- \log \sigma_i)
\tag{2.44}
$$

여기서 $\mathbf{z} = g(\boldsymbol{\epsilon}, \boldsymbol{\phi}, \mathbf{x})$ 이다.

---

### 2.5.1 완전 공분산 가우시안 사후분포 (Full-covariance Gaussian posterior)

분리된(factorized) 가우시안 사후분포는  
완전 공분산(full covariance)을 가지는 가우시안으로 확장될 수 있다.

$$
q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})
= \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}, \boldsymbol{\Sigma})
\tag{2.45}
$$

이 분포의 재매개변수화(reparameterization)는 다음과 같이 주어진다.

$$
\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)
\tag{2.46}
$$

$$
\mathbf{z} = \boldsymbol{\mu} + \mathbf{L}\boldsymbol{\epsilon}
\tag{2.47}
$$

여기서 $\mathbf{L}$ 은 대각 원소(diagonal element)가 0이 아닌 하삼각(lower) 또는 상삼각(upper) 행렬이다.  
대각 이외의 원소(off-diagonal element)는 $\mathbf{z}$ 의 각 성분 간의 상관관계(covariance)를 정의한다.

완전 공분산 가우시안에 대해 이러한 매개변수화를 사용하는 이유는  
야코비안의 행렬식(Jacobian determinant)이 매우 단순해지기 때문이다.  
이 경우 야코비안은 다음과 같다.

$$
\frac{\partial \mathbf{z}}{\partial \boldsymbol{\epsilon}} = \mathbf{L}
$$

삼각행렬(triangular matrix)의 행렬식은 그 대각 원소들의 곱이 된다는 점에 유의하라.  
따라서 이 매개변수화에서는 다음이 성립한다.

$$
\log \left|\det\!\left(\frac{\partial \mathbf{z}}{\partial \boldsymbol{\epsilon}}\right)\right|
= \sum_i \log |L_{ii}|
\tag{2.48}
$$

> (1) 식 (2.45)의 의미  
> 이 식은 분리된(factorized) 가우시안 사후분포를  
> 완전 공분산(full-covariance)을 가지는 형태로 확장한 것이다.  
> 즉, $\boldsymbol{\Sigma}$ 가 대각행렬(diagonal matrix)이 아닌,  
> 모든 성분 간의 공분산(covariance)을 포함하는 일반적인 행렬이 된다.  
> 이로써 $\mathbf{z}$ 의 각 차원이 서로 독립(independent)하지 않고  
> 상관관계(correlation)를 가질 수 있게 된다.  
>
> (2) 식 (2.47)의 의미  
> 재매개변수화 $\mathbf{z} = \boldsymbol{\mu} + \mathbf{L}\boldsymbol{\epsilon}$ 에서  
> 행렬 $\mathbf{L}$ 은 공분산 행렬 $\boldsymbol{\Sigma}$ 의 분해(cholesky decomposition)를 통해 얻어진다.  
> 즉, $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$ 으로 나타낼 수 있으며,  
> $\mathbf{L}$ 은 일반적으로 **하삼각(lower triangular) 행렬**이다.  
> 이렇게 하면 샘플링 과정에서 $\boldsymbol{\epsilon}$ (표준 정규분포에서 추출된 노이즈)에  
> 선형 변환을 적용하여 공분산 구조를 반영할 수 있다.  
>
> (3) 삼각행렬(triangular matrix)의 예시  
> 예를 들어, 3차원 잠재변수의 경우 $\mathbf{L}$ 이 하삼각 행렬이면 다음과 같다.
>
> $$
> \mathbf{L} =
> \begin{bmatrix}
> L_{11} & 0 & 0 \\
> L_{21} & L_{22} & 0 \\
> L_{31} & L_{32} & L_{33}
> \end{bmatrix}
> $$
>
> 여기서:
> - 대각 원소 $L_{11}, L_{22}, L_{33}$ 는 각 차원의 분산(scale)을 조절하고,  
> - 비대각 원소(off-diagonal) $L_{21}, L_{31}, L_{32}$ 는  
>   차원 간의 공분산(covariance) 또는 상관성(correlation)을 반영한다.  
>
> (4) 삼각행렬의 행렬식(det)의 계산  
> 하삼각행렬의 행렬식은 대각 원소들의 곱으로 단순하게 계산된다.
>
> $$
> \det(\mathbf{L}) = L_{11} \cdot L_{22} \cdot L_{33}
> $$
>
> 따라서 로그를 취하면 다음과 같은 매우 간단한 형태가 된다.
>
> $$
> \log|\det(\mathbf{L})| = \sum_i \log |L_{ii}|
> $$

---

사후분포의 로그밀도(log-density)는 다음과 같다.

$$
\log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})
= \log p(\boldsymbol{\epsilon}) - \sum_i \log |L_{ii}|
\tag{2.49}
$$

이 매개변수화(parameterization)는  
$\mathbf{z}$ 의 공분산 행렬 $\boldsymbol{\Sigma}$에 대한 Cholesky 분해 $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^\top$ 에 대응한다.

$$
\boldsymbol{\Sigma}
= \mathbb{E}\!\left[(\mathbf{z} - \mathbb{E}[\mathbf{z}])(\mathbf{z} - \mathbb{E}[\mathbf{z}])^\top\right]
\tag{2.50}
$$

$$
= \mathbb{E}\!\left[\mathbf{L}\boldsymbol{\epsilon}(\mathbf{L}\boldsymbol{\epsilon})^\top\right]
= \mathbf{L}\,\mathbb{E}[\boldsymbol{\epsilon}\boldsymbol{\epsilon}^\top]\,\mathbf{L}^\top
\tag{2.51}
$$

$$
= \mathbf{L}\mathbf{L}^\top
\tag{2.52}
$$

$\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$이므로 $\mathbb{E}[\boldsymbol{\epsilon}\boldsymbol{\epsilon}^\top] = I$ 임에 유의하라.

> (1) 식 (2.50)의 의미  
> 공분산 행렬 $\boldsymbol{\Sigma}$ 는 잠재변수 $\mathbf{z}$ 의 중심화된 값 $(\mathbf{z}-\mathbb{E}[\mathbf{z}])$의 외적을 평균낸 기댓값으로 정의된다.  
> 즉 $\boldsymbol{\Sigma}=\mathbb{E}\\left[(\mathbf{z}-\mathbb{E}[\mathbf{z}])(\mathbf{z}-\mathbb{E}[\mathbf{z}])^{\top}\right]$ 이다.
>
> (2) 식 (2.51)의 의미  
> $\mathbf{z}=\boldsymbol{\mu}+\mathbf{L}\boldsymbol{\epsilon}$ 을 대입하면 중심화된 값은 $\mathbf{z}-\mathbb{E}[\mathbf{z}]=\mathbf{L}\boldsymbol{\epsilon}$ 이다.  
> 따라서 $\boldsymbol{\Sigma}=\mathbb{E}\\left[\mathbf{L}\boldsymbol{\epsilon}(\mathbf{L}\boldsymbol{\epsilon})^{\top}\right]=\mathbf{L}\,\mathbb{E}[\boldsymbol{\epsilon}\boldsymbol{\epsilon}^{\top}]\,\mathbf{L}^{\top}$ 로 전개된다.
>
> (3) 식 (2.52)의 의미  
> $\boldsymbol{\epsilon}\sim\mathcal{N}(0,I)$ 이므로 $\mathbb{E}[\boldsymbol{\epsilon}\boldsymbol{\epsilon}^{\top}]=I$ 이다.  
> 결과적으로 $\boldsymbol{\Sigma}=\mathbf{L}\mathbf{L}^{\top}$ 가 되며,  
> 이는 공분산 행렬의 촐레스키 분해(Cholesky decomposition)에 해당한다.

---

원하는 성질,  
즉 삼각행렬성과 0이 아닌 대각 원소를 가지는 행렬 $\mathbf{L}$ 을 구성하는 한 가지 방법은 다음과 같다:

$$
(\boldsymbol{\mu}, \log \boldsymbol{\sigma}, \mathbf{L}')
\leftarrow \mathrm{EncoderNeuralNet}_{\boldsymbol{\phi}}(\mathbf{x})
\tag{2.53}
$$

$$
\mathbf{L} \leftarrow \mathbf{L}_{\text{mask}} \odot \mathbf{L}' + \mathrm{diag}(\boldsymbol{\sigma})
\tag{2.54}
$$

이후 앞서 설명한 것처럼 $\mathbf{z} = \boldsymbol{\mu} + \mathbf{L}\boldsymbol{\epsilon}$ 으로 계산을 진행한다.  

여기서 $\mathbf{L}_{\text{mask}}$ 는 마스크 행렬로,  
대각선 위와 대각선에는 0을, 대각선 아래에는 1을 가지도록 구성된다.  
즉, $\mathbf{L}'$ 의 상삼각 부분은 제거되고 하삼각 부분만 남게 된다.  

> (1) 식 (2.53)의 의미  
> 인코더 신경망은 입력 $\mathbf{x}$ 로부터  
> 평균 $\boldsymbol{\mu}$, 로그표준편차 $\log\boldsymbol{\sigma}$, 그리고 공분산 구조를 학습하기 위한 가중치 행렬 $\mathbf{L}'$ 을 출력한다.  
> 여기서 $\boldsymbol{\sigma}$ 는 각 잠재변수의 단변량 표준편차(대각 항)를,  
> $\mathbf{L}'$ 은 변수 간 상관관계(비대각 항)를 학습한다.  
> 다만 $\mathbf{L}'$ 은 아직 삼각행렬 형태를 보장하지 않는다.
>
> (2) 식 (2.54)의 의미  
> 마스크 행렬 $\mathbf{L}_{\text{mask}}$ 를 사용해 $\mathbf{L}'$ 의 상삼각 원소를 제거하고, 대각에는 $\boldsymbol{\sigma}$ 를 더하여  
> $\mathbf{L}$ 이 0이 아닌 대각 원소를 가진 하삼각 행렬이 되도록 만든다.  
> 즉, $\mathbf{L}'$ 로부터 공분산 구조를 반영하면서도 각 변수의 분산 크기(σ)를 동시에 반영한다.

마스킹 처리 덕분에, 야코비안 행렬 $\dfrac{\partial \mathbf{z}}{\partial \boldsymbol{\epsilon}}$은 $\boldsymbol{\sigma}$ 값이 대각 원소로 놓인 삼각행렬이 된다.  
따라서 로그 행렬식(log-determinant)은 분리된 가우시안 경우와 동일하게 다음과 같이 계산된다.

$$
\log \left|\det\!\left(\frac{\partial \mathbf{z}}{\partial \boldsymbol{\epsilon}}\right)\right|
= \sum_i \log \sigma_i
\tag{2.55}
$$

좀 더 일반적으로,  
$\mathbf{z} = \mathbf{L}\boldsymbol{\epsilon} + \boldsymbol{\mu}$를 (미분 가능하고 비선형적인) 변환들의 연쇄(chain)로 대체할 수 있다.  

이때, 연쇄의 각 단계에서의 야코비안(Jacobian)이  
비영(非零) 대각 원소를 가진 삼각행렬(triangular matrix) 형태라면,  
로그 행렬식(log determinant)은 여전히 단순하게 계산된다.  

이 원리는 Inverse Autoregressive Flow (IAF)에서 사용되며,  
이는 *Kingma et al., 2016* 에서 처음 제안되었고 이 책의 3장에서 자세히 논의된다.

---

**알고리즘 2. 단일 데이터 포인트 ELBO의 비편향 추정 (Unbiased estimate of single-datapoint ELBO)**  

이 알고리즘은 완전 공분산(full-covariance) 가우시안 추론 모델과  
분리된 베르누이(factorized Bernoulli) 생성 모델을 가지는 예시 VAE의  
단일 데이터 포인트(single-datapoint)에 대한 ELBO를 비편향적으로 추정한다.  

여기서 $\mathbf{L}_{\text{mask}}$ 는  
대각선 위와 대각선에 0을, 대각선 아래에는 1을 가지는 마스크 행렬(masking matrix)이다.  

---

**데이터 (Data):**  
- $\mathbf{x}$: 데이터 포인트 (필요시 조건부 정보 포함 가능)  
- $\boldsymbol{\epsilon}$: $p(\boldsymbol{\epsilon}) = \mathcal{N}(0, I)$ 로부터의 무작위 샘플  
- $\boldsymbol{\theta}$: 생성 모델(generative model) 파라미터  
- $\boldsymbol{\phi}$: 추론 모델(inference model) 파라미터  
- $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$: 추론 모델 (Inference model)  
- $p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})$: 생성 모델 (Generative model)  

**결과 (Result):**  
- $\tilde{\mathcal{L}}$: 단일 데이터 포인트에 대한 ELBO의 비편향 추정치  

---

**절차 (Procedure):**

$$
(\boldsymbol{\mu}, \log \boldsymbol{\sigma}, \mathbf{L}')
\leftarrow \mathrm{EncoderNeuralNet}_{\boldsymbol{\phi}}(\mathbf{x})
$$

$$
\mathbf{L}
\leftarrow \mathbf{L}_{\text{mask}} \odot \mathbf{L}' + \mathrm{diag}(\boldsymbol{\sigma})
$$

$$
\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)
$$

$$
\mathbf{z} = \mathbf{L}\boldsymbol{\epsilon} + \boldsymbol{\mu}
$$

$$
\tilde{\mathcal{L}}_{\log q_{\mathbf{z}}} 
\leftarrow -\sum_i \left(\tfrac{1}{2}(\epsilon_i^2 + \log(2\pi)) + \log \sigma_i \right)
\quad \text{for } q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})
$$

$$
\tilde{\mathcal{L}}_{\log p_{\mathbf{z}}} 
\leftarrow -\sum_i \left(\tfrac{1}{2}(z_i^2 + \log(2\pi))\right)
\quad \text{for } p_{\boldsymbol{\theta}}(\mathbf{z})
$$

$$
\mathbf{p} \leftarrow \mathrm{DecoderNeuralNet}_{\boldsymbol{\theta}}(\mathbf{z})
$$

$$
\tilde{\mathcal{L}}_{\log p_{\mathbf{x}}}
\leftarrow \sum_i \left(x_i \log p_i + (1 - x_i)\log(1 - p_i)\right)
\quad \text{for } p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})
$$

$$
\tilde{\mathcal{L}} 
= \tilde{\mathcal{L}}_{\log p_{\mathbf{x}}}
+ \tilde{\mathcal{L}}_{\log p_{\mathbf{z}}}
- \tilde{\mathcal{L}}_{\log q_{\mathbf{z}}}
$$

---

> (1) 인코더 단계  
>    $\mathrm{EncoderNeuralNet}_{\boldsymbol{\phi}}(\mathbf{x})$는  
>    입력 $\mathbf{x}$ 로부터 평균 $\boldsymbol{\mu}$, 로그표준편차 $\log\boldsymbol{\sigma}$, 그리고 초기 공분산 행렬 $\mathbf{L}'$ 을 추정한다.  
>
> (2) 마스크 적용 단계  
>    마스크 행렬 $\mathbf{L}_{\text{mask}}$ 를 적용해 상삼각 항을 제거하고, 대각에는 $\boldsymbol{\sigma}$를 추가하여  
>   하삼각 형태의 공분산 행렬 $\mathbf{L}$을 생성한다.  
>
> (3) 잠재변수 샘플링  
>    $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$에 따라 노이즈를 샘플링한 뒤, $\mathbf{z} = \mathbf{L}\boldsymbol{\epsilon} + \boldsymbol{\mu}$ 로 재매개변수화한다.  
>
> (4) ELBO 항 계산  
>    - $$\tilde{\mathcal{L}}_{\log q_{\mathbf{z}}}$$: 인코더의 사후 분포 항  
>    - $$\tilde{\mathcal{L}}_{\log p_{\mathbf{z}}}$$: 잠재변수의 사전 분포 항  
>    - $$\tilde{\mathcal{L}}_{\log p_{\mathbf{x}}}$$: 복원된 입력의 로그우도 항  
>
> (5) ELBO 결합  
>    최종 ELBO 추정치는 다음과 같다.  
>
>    $$
>    \tilde{\mathcal{L}} = 
>    \tilde{\mathcal{L}}_{\log p_{\mathbf{x}}}
>    + \tilde{\mathcal{L}}_{\log p_{\mathbf{z}}}
>    - \tilde{\mathcal{L}}_{\log q_{\mathbf{z}}}
>    $$  
>
>    이 식은 단일 데이터 포인트에 대한 ELBO의 비편향 추정치로 사용된다.

---

## 2.6 주변가능도(Marginal Likelihood)의 추정

VAE를 학습한 후에는,  
Rezende 등(2014)에 의해 처음 제안된 중요도 샘플링(importance sampling) 기법을 사용하여  
모델 하에서 데이터의 확률을 추정할 수 있다.

> 중요도 샘플링(importance sampling)은  
> 직접 샘플링하기 어려운 분포 $p(\mathbf{z})$ 하에서 함수 $f(\mathbf{z})$의 기대값 $\mathbb{E}_{p}[f(\mathbf{z})]$을  
> 샘플링이 쉬운 제안 분포 $q(\mathbf{z})$로부터 얻은 샘플로 근사하는 방법이다.  
>
> 먼저 $q(\mathbf{z})$에서 샘플 $\mathbf{z}^{(1)}, \dots, \mathbf{z}^{(L)} \sim q(\mathbf{z})$를 추출하고,  
> 각 샘플에 중요도 가중치 $w^{(l)} = \frac{p(\mathbf{z}^{(l)})}{q(\mathbf{z}^{(l)})}$를 곱해  
> 기대값을 다음과 같이 추정한다.
>
> $$
> \hat{\mu} = \frac{1}{L}\sum_{l=1}^{L} w^{(l)} f(\mathbf{z}^{(l)}),
> \quad \text{또는} \quad
> \hat{\mu} = \frac{\sum_{l=1}^{L} w^{(l)} f(\mathbf{z}^{(l)})}{\sum_{l=1}^{L} w^{(l)}}.
> $$
>
> 직관적으로, $q$에서 너무 자주 뽑힌 영역은 가중치로 덜어 주고,  
> $q$에 비해 $p$가 큰 영역은 더 큰 가중치를 부여해 균형을 맞춘다.  
>
> 이 방법은 $q$가 $p$의 높은 확률질량 영역을 잘 커버할 때 잘 작동하지만,  
> 두 분포가 겹치지 않으면 가중치의 분산이 커져 추정이 불안정해진다.  
> 특히 고차원 공간에서는 이런 문제가 심각해지므로,  
> $q$의 품질을 개선하거나 샘플 수를 늘리는 것이 필요하다.  
>
> VAE에서는 주변가능도 추정 시  
> 가중치 $w^{(l)} = \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}^{(l)})}{q_{\boldsymbol{\phi}}(\mathbf{z}^{(l)}|\mathbf{x})}$ 형태로 등장하며,  
> IWAE는 이 가중치들을 여러 개 사용해 로그 평균을 취함으로써  
> ELBO보다 더 타이트한 하한을 얻는다.  
>
> 쉽게 말해, 목표 분포 $p$의 “지형”을 직접 탐색하기 어려울 때,  
> 더 다루기 쉬운 분포 $q$를 따라가며  
> 각 지점이 실제 지형 $p$에서 얼마나 중요한지를 가중치로 보정하는 과정이  
> 바로 중요도 샘플링이다.

데이터 포인트 하나의 주변가능도(marginal likelihood)는 다음과 같이 쓸 수 있다.

$$
\log p_{\boldsymbol{\theta}}(\mathbf{x})
= \log
\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}
\left[
\frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}
{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}
\right]
\tag{2.56}
$$

> 주변가능도 $p_{\boldsymbol{\theta}}(\mathbf{x}) = \int p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})\,d\mathbf{z}$는  
> 적분 안에 추론 분포 $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$를 곱하고 나누어  
> 기대값 형태로 바꿀 수 있다.  
>
> $$
> p_{\boldsymbol{\theta}}(\mathbf{x})
> = \int q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})
> \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}
> {q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}\,d\mathbf{z}
> = \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}
> \!\left[\frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}
> {q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}\right].
> $$
>
> 양변에 로그를 취하면 식 (2.56)이 되며,  
> 이는 적분을 기대값으로 표현한 항등식으로,  
> 주변가능도를 샘플링 기반으로 근사할 수 있게 해준다.  
> 이러한 표현은 중요도 샘플링(importance sampling)의 기초가 된다.

$q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ 로부터 무작위 샘플을 여러 개 추출하면,  
이를 이용한 몬테카를로 근사(Monte Carlo estimator)는 다음과 같다.

$$
\log p_{\boldsymbol{\theta}}(\mathbf{x})
\approx
\log
\frac{1}{L}
\sum_{l=1}^{L}
\frac{
p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}^{(l)})
}{
q_{\boldsymbol{\phi}}(\mathbf{z}^{(l)}\mid\mathbf{x})
}
\tag{2.57}
$$

여기서 각 $\mathbf{z}^{(l)} \sim q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$는 추론 모델로부터 샘플링된 잠재변수이다.  

$L$ 을 크게 할수록 근사는 주변가능도의 더 정확한 추정값에 가까워진다.  

사실상, $L \to \infty$ 일 때 이 몬테카를로 추정치는  
실제 주변가능도(marginal likelihood)에 수렴하게 된다.

---

$L = 1$ 로 설정하면, 이는 VAE의 ELBO 추정량과 동일하다.  

또한 식 (2.57)의 추정량을 목적함수(objective function)로 사용할 수도 있는데,  
이는 Importance Weighted Autoencoders (IWAE) (Burda et al., 2015)에서 사용된 목적함수이다.  

해당 연구에서는 $L$ 값이 커질수록 목적함수가 더 "tight"해진다고 보였다.  
(즉, 진짜 로그가능도 $\log p_{\boldsymbol{\theta}}(\mathbf{x})$ 에 더 근접한다고)   

이후 Cremer et al. (2017)은 IWAE의 목적함수가  
특정한 추론 모델 형태를 갖는 ELBO 목적함수로 재해석될 수 있다는 사실을 밝혔다.  

그러나 이러한 “더 타이트한 하한”을 최적화하는 접근 방식에는 단점이 있다.  
즉, 중요도 가중 추정(importance weighted estimate)은  
고차원 잠재공간에서 스케일링이 매우 비효율적이라는 문제가 있다.

> $L=1$일 때는 샘플을 하나만 사용하므로,  
> 이는 일반적인 VAE의 ELBO 계산과 완전히 동일하다.  
> 하지만 $L$을 늘리면 샘플을 여러 개 사용해  
> 주변가능도를 더 정밀하게 추정할 수 있고,  
> 그 결과 목적함수가 실제 로그가능도에 더 가까워진다.  
> 이것이 IWAE의 핵심 아이디어다.  
> 다만, 샘플 수를 늘리면 계산 비용이 커지고,  
> 고차원 잠재공간에서는 중요도 가중치의 분산이 매우 커져  
> 학습 효율이 급격히 떨어지는 한계가 있다.

---

## 2.7 주변가능도(Marginal Likelihood)와 KL 발산으로서의 ELBO

ELBO의 잠재적인 타이트니스(tightness)를 향상시키는 한 가지 방법은 생성 모델의 유연성을 증가시키는 것이다.  

이것은 ELBO와 KL 발산 사이의 연결을 통해 이해될 수 있다.

> ELBO를 더 타이트하게 만들려면 단순히 중요도 샘플을 늘리는 대신,  
> 생성 모델 자체의 표현력을 높여 데이터 분포를 더 잘 설명하도록 하는 것이 효과적이다.  
> 이는 ELBO가 결국 데이터 분포와 모델 분포 간의 KL 발산 최소화 문제로 연결되기 때문이다.

---

독립적이고 동일분포(i.i.d.)를 따르는 크기 $N_D$ 의 데이터셋 $\mathcal{D}$ 에 대해,  
최대우도 기준(criterion)은 다음과 같다.

$$
\begin{align}
\log p_{\boldsymbol{\theta}}(\mathcal{D})
&= \frac{1}{N_D} \sum_{\mathbf{x} \in \mathcal{D}} \log p_{\boldsymbol{\theta}}(\mathbf{x})
\tag{2.58} \\[6pt]
&= \mathbb{E}_{q_{\mathcal{D}}(\mathbf{x})} [\log p_{\boldsymbol{\theta}}(\mathbf{x})]
\tag{2.59}
\end{align}
$$

여기서 $q_{\mathcal{D}}(\mathbf{x})$ 는 혼합 분포인 경험적(데이터) 분포이다.

$$
\begin{align}
q_{\mathcal{D}}(\mathbf{x})
&= \frac{1}{N} \sum_{i=1}^{N} q_{\mathcal{D}}^{(i)}(\mathbf{x})
\tag{2.60}
\end{align}
$$

각 성분 $q_{\mathcal{D}}^{(i)}(\mathbf{x})$는 일반적으로  
연속형 데이터의 경우 $\mathbf{x}^{(i)}$ 값에 중심을 둔 디랙 델타(Dirac delta) 분포에 대응하며,  
이산형 데이터의 경우 $\mathbf{x}^{(i)}$ 값에 모든 확률 질량이 집중된 이산 분포에 대응한다.  

> 경험적 분포 $q_{\mathcal{D}}(\mathbf{x})$는 데이터셋을 확률적으로 표현한 분포로,  
> 각 데이터 포인트가  
> “그 지점에만 확률이 몰린 아주 뾰족한 분포(디랙 델타)”로 표현된다고 볼 수 있다.  
> 쉽게 말해, 모든 데이터 샘플에 동일한 가중치를 주어 평균낸 “데이터의 실제 분포”이다.

데이터 분포와 모델 분포 사이의 쿨백-라이블러(Kullback-Leibler, KL) 발산은  
음의 로그우도(negative log-likelihood)에 상수를 더한 형태로 다시 쓸 수 있다.

$$
\begin{align}
D_{\mathrm{KL}}(q_{\mathcal{D}}(\mathbf{x}) \| p_{\boldsymbol{\theta}}(\mathbf{x}))
&= - \mathbb{E}_{q_{\mathcal{D}}(\mathbf{x})} [\log p_{\boldsymbol{\theta}}(\mathbf{x})]
+ \mathbb{E}_{q_{\mathcal{D}}(\mathbf{x})} [\log q_{\mathcal{D}}(\mathbf{x})]
\tag{2.61} \\[6pt]
&= -\log p_{\boldsymbol{\theta}}(\mathcal{D}) + \text{constant}
\tag{2.62}
\end{align}
$$

여기서 상수항은 $-\mathcal{H}(q_{\mathcal{D}}(\mathbf{x}))$ 이다.  

> 엔트로피(entropy)는 확률분포 $q_{\mathcal{D}}(\mathbf{x})$의 “평균적인 놀람 정도”를 의미한다.  
> 여기서 자기 정보량(self-information)은 어떤 사건이 일어났을 때의 놀람 정도로,  
> $-\log q_{\mathcal{D}}(\mathbf{x})$로 정의된다.  
> 따라서 엔트로피는 이 놀람의 평균값, 즉 $$\mathcal{H}(q_{\mathcal{D}}(\mathbf{x})) = - \mathbb{E}_{q_{\mathcal{D}}(\mathbf{x})}[\log q_{\mathcal{D}}(\mathbf{x})]$$ 로 표현된다.  
> KL 발산 식의 두 번째 항이 이 정의와 부호만 반대이므로 상수항을 $$-\mathcal{H}(q_{\mathcal{D}}(\mathbf{x}))$$로 쓸 수 있다.

따라서 위의 KL 발산을 최소화하는 것은 데이터 로그가능도 $\log p_{\boldsymbol{\theta}}(\mathcal{D})$를 최대화하는 것과 동일하다.

---

경험적 데이터 분포 $q_{\mathcal{D}}(\mathbf{x})$와 추론 모델을 결합하면,  
데이터 $\mathbf{x}$ 와 잠재변수 $\mathbf{z}$ 에 대한 결합분포(joint distribution)를 다음과 같이 얻을 수 있다.

$$
q_{\mathcal{D}, \boldsymbol{\phi}}(\mathbf{x}, \mathbf{z})
= q_{\mathcal{D}}(\mathbf{x}) q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})
$$

$q_{\mathcal{D}, \boldsymbol{\phi}}(\mathbf{x}, \mathbf{z})$ 와 $p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})$ 사이의 KL 발산은  
음의 ELBO(negative ELBO)에 상수(constant)를 더한 형태로 쓸 수 있다.

$$
\begin{align}
& D_{KL}\!\big(q_{\mathcal{D},\boldsymbol{\phi}}(\mathbf{x},\mathbf{z}) \,\|\, p_{\boldsymbol{\theta}}(\mathbf{x},\mathbf{z})\big) \tag{2.63} \\[6pt]
&= -\,\mathbb{E}_{q_{\mathcal{D}}(\mathbf{x})}
\!\left[
\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid \mathbf{x})}
\big[\log p_{\boldsymbol{\theta}}(\mathbf{x},\mathbf{z})
- \log q_{\boldsymbol{\phi}}(\mathbf{z}\mid \mathbf{x})\big]
- \log q_{\mathcal{D}}(\mathbf{x})
\right] \tag{2.64} \\[6pt]
&= -\,\mathcal{L}_{\boldsymbol{\theta},\boldsymbol{\phi}}(\mathcal{D}) \;+\; \text{constant} \tag{2.65}
\end{align}
$$

> (2.63)에서 시작: $$D_{KL}(q\|p)=\mathbb{E}_{q(\mathbf{x},\mathbf{z})}[\log q(\mathbf{x},\mathbf{z})-\log p(\mathbf{x},\mathbf{z})]$$  
> 여기서 $$q(\mathbf{x},\mathbf{z})=q_{\mathcal{D}}(\mathbf{x})\,q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$$이므로  
> $$\log q(\mathbf{x},\mathbf{z})=\log q_{\mathcal{D}}(\mathbf{x})+\log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$$  
> 
> 기대값의 사슬 법칙으로 분해:  
> $$\mathbb{E}_{q(\mathbf{x},\mathbf{z})}[\,\cdot\,] =\mathbb{E}_{q_{\mathcal{D}}(\mathbf{x})}\!\big[\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}[\,\cdot\,]\big]$$  
> 
> 대입/정리:  
> 
> $$
> \begin{aligned}
> D_{KL}
> &=\mathbb{E}_{q_{\mathcal{D}}(\mathbf{x})}\!\Big[
> \log q_{\mathcal{D}}(\mathbf{x})
> +\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}\![\log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})-\log p_{\boldsymbol{\theta}}(\mathbf{x},\mathbf{z})]
> \Big] \\
> &=-\,\mathbb{E}_{q_{\mathcal{D}}(\mathbf{x})}\!\Big[
> \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}\![\log p_{\boldsymbol{\theta}}(\mathbf{x},\mathbf{z})-\log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})]
> -\log q_{\mathcal{D}}(\mathbf{x})
> \Big],
> \end{aligned}
> $$
>  
> 이것이 (2.64)

여기서 상수항은 $-\mathcal{H}(q_{\mathcal{D}}(\mathbf{x}))$ 이다.  
따라서 ELBO를 최대화하는 것은  
이 KL 발산 $D_{KL}(q_{\mathcal{D}, \boldsymbol{\phi}}(\mathbf{x}, \mathbf{z}) \| p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}))$을 최소화하는 것과 동치이다.

이제 최대우도(ML)와 ELBO의 관계는 다음의 간단한 식으로 요약된다.

$$
\begin{align}
& D_{KL}\!\big(q_{\mathcal{D},\boldsymbol{\phi}}(\mathbf{x},\mathbf{z}) \,\|\, p_{\boldsymbol{\theta}}(\mathbf{x},\mathbf{z})\big) \tag{2.66} \\[6pt]
&= D_{KL}\!\big(q_{\mathcal{D}}(\mathbf{x}) \,\|\, p_{\boldsymbol{\theta}}(\mathbf{x})\big)
+ \mathbb{E}_{q_{\mathcal{D}}(\mathbf{x})}
\!\Big[
D_{KL}\!\big(q_{\mathcal{D},\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})
\,\|\, p_{\boldsymbol{\theta}}(\mathbf{z}\mid\mathbf{x})\big)
\Big] \tag{2.67} \\[6pt]
&\ge D_{KL}\!\big(q_{\mathcal{D}}(\mathbf{x}) \,\|\, p_{\boldsymbol{\theta}}(\mathbf{x})\big) \tag{2.68}
\end{align}
$$

> (2.66) → (2.67):  
> $D_{KL}(q(x,z)\|p(x,z)) = \mathbb{E}_{q(x,z)}[\log q(x,z) - \log p(x,z)]$에서  
> $q(x,z)=q(x)q(z|x)$, $p(x,z)=p(x)p(z|x)$를 대입하면  
>
> $$
> \begin{aligned}
> D_{KL}(q(x,z)\|p(x,z))
> &= \mathbb{E}_{q(x,z)}[\log q(x) - \log p(x)]
> + \mathbb{E}_{q(x,z)}[\log q(z|x) - \log p(z|x)] \\[4pt]
> &= \mathbb{E}_{q(x)}[\log q(x) - \log p(x)]
> + \mathbb{E}_{q(x)}\!\Big[\mathbb{E}_{q(z|x)}[\log q(z|x) - \log p(z|x)]\Big].
> \end{aligned}
> $$  
>
> 여기서 첫 번째 항의 기대값 기준이 $q(x,z)$에서 $q(x)$로 바뀌는 이유는  
> $\log q(x)-\log p(x)$가 $z$에 의존하지 않기 때문이다.  
>
> (2.67) → (2.68):  
> 조건부 KL 발산 $D_{KL}(q(z|x)\|p(z|x))$는 항상 0 이상이므로,  
> 두 번째 항이 제거되면 전체 KL 발산의 하한이 된다.  
> 따라서
>
> $$
> D_{KL}(q(x,z)\|p(x,z)) \ge D_{KL}(q(x)\|p(x))
> $$
>
> 이 되어 식 (2.68)이 성립한다.

---

하나의 추가적인 관점은, ELBO를 증강된 공간(augmented space)에서의  
최대우도 목표(maximum likelihood objective)로 볼 수 있다는 것이다.  

인코더 $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$가 고정되어 있다고 하면, 결합분포 $p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})$는  
원래의 데이터 $\mathbf{x}$와 각 데이터 포인트에 연관된  
(확률적) 보조 특징(stochastic auxiliary features) $\mathbf{z}$에 대한  
증강된 경험적 분포(augmented empirical distribution)로 볼 수 있다.  

> 여기서 말하는 “증강된 공간(augmented space)”이란  
> 원래 데이터 $\mathbf{x}$만 다루는 대신,  
> 그에 대응하는 잠재변수(또는 보조 변수) $\mathbf{z}$까지 포함한  
> 확장된 공간 $(\mathbf{x}, \mathbf{z})$를 의미한다.  
> 즉, ELBO를 단순히 데이터 $\mathbf{x}$의 가능도를 근사하는 식이 아니라,  
> “데이터와 그에 대응하는 잠재표현 전체에 대한 최대우도 학습”으로  
> 해석할 수 있다는 뜻이다.  
> 이런 관점에서는 인코더 $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$가  
> 데이터마다 확률적으로 보조 특징 $\mathbf{z}$를 부여하는 역할을 하고,  
> 생성모델 $p_{\boldsymbol{\theta}}(\mathbf{x},\mathbf{z})$는  
> 이 결합된 데이터–잠재공간을 최대우도 방식으로 학습한다고 볼 수 있다.

즉, 모델 $p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})$는  
원래 데이터와 그에 대응하는 보조 특징 모두에 대한 결합 모델(joint model)을 정의한다.  
(그림 2.4 참조)

---

그림 2.4:  
최대우도(ML, Maximum Likelihood) 목표는 $D_{KL}(q_{\mathcal{D},\boldsymbol{\phi}}(\mathbf{x}) \,\|\, p_{\boldsymbol{\theta}}(\mathbf{x}))$의 최소화로 볼 수 있다.  

반면 ELBO 목표는 $D_{KL}(q_{\mathcal{D},\boldsymbol{\phi}}(\mathbf{x}, \mathbf{z}) \,\|\, p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}))$의 최소화로 볼 수 있으며,  
이는 $D_{KL}(q_{\mathcal{D},\boldsymbol{\phi}}(\mathbf{x}) \,\|\, p_{\boldsymbol{\theta}}(\mathbf{x}))$보다 항상 크거나 같다(즉, 그 값을 위에서 제한한다).

완벽한 적합(perfect fit)이 불가능할 경우,  
KL 발산의 방향성 때문에 $p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})$는 $q_{\mathcal{D},\boldsymbol{\phi}}(\mathbf{x}, \mathbf{z})$보다 일반적으로 더 큰 분산(variance)을 갖게 된다.

> KL 발산의 정의는  
>
> $$
> D_{KL}(q\|p)=\mathbb{E}_{q}[\log q(x,z)-\log p(x,z)]
> $$
>
> 이다.  
> 이 식은 $q(x,z)$가 큰 영역(즉, 실제 데이터가 자주 등장하는 곳)에서  
> $p(x,z)$가 너무 작으면 $\log p(x,z)$ 항이 크게 음수가 되어 전체 KL 값이 급격히 커진다.  
>  
> 반대로 $p(x,z)$가 $q(x,z)$보다 더 넓게 퍼져 있더라도,  
> 즉 데이터가 거의 없는 영역에 확률 질량을 조금 더 주더라도 KL 값에는 큰 영향이 없다.  
>  
> 따라서 $D_{KL}(q\|p)$를 최소화할 때 모델 분포 $p_{\boldsymbol{\theta}}(x,z)$는  
> “데이터가 있는 영역에서 확률이 너무 작아지지 않도록” 학습되며,  
> 그 결과 데이터 영역을 넉넉히 덮기 위해 전체적으로 분산이 커지는 경향을 보이게 된다.

<img src="/assets/img/books/intro-to-vae/2/image_4.png" alt="image" width="720px"> 

---

