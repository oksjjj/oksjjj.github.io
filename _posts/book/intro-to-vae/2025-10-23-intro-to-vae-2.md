---
layout: post
title: "[1. Variational Autoencoders] An Introduction to Variational Autoencoders"
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
그러한 모델에서 로그우도(log-likelihood)와 사후분포(posterior distribution)를 추정하는 문제를 소개하였다.  

변분 오토인코더(VAE) 프레임워크는  
확률적 경사하강법(SGD, Stochastic Gradient Descent)을 이용하여  
DLVM과 이에 대응하는 추론 모델(inference model)을  
동시에 최적화할 수 있는 계산 효율적인 방법을 제공한다.

---

DLVM의 계산 불가능한(intractable) 사후 추론과 학습 문제를 계산 가능한 형태로 바꾸기 위해,  
매개변수를 갖는 추론 모델(parametric inference model) $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$를 도입한다.  

이 모델은 인코더(encoder) 또는 인식 모델(recognition model)이라고도 불린다.  
여기서 $\boldsymbol{\phi}$는 이 추론 모델의 매개변수(parameter)를 의미하며, 변분 매개변수(variational parameters)라고 부른다.  

이 변분 매개변수 $\boldsymbol{\phi}$를 최적화하여 다음이 성립하도록 한다:

$$
q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \approx p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})
\tag{2.1}
$$

이후 설명하겠지만, 이러한 사후분포의 근사는 주변우도(marginal likelihood)를 효율적으로 최적화하는 데 도움이 된다.

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

반면, VAE에서는 데이터 포인트 전체에 대해 하나의 공통된 변분 매개변수를 공유하는 전략을 사용한다.  
이 방식을 상각 변분 추론(amortized variational inference) (Gershman & Goodman, 2014)이라 부른다.  

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

식 (2.8)의 첫 번째 항은 변분 하한(variational lower bound), 즉 ELBO (Evidence Lower Bound)라고 부른다.

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

## 2.3 확률적 경사 기반 ELBO 최적화 (Stochastic Gradient-Based Optimization of the ELBO)  

ELBO의 중요한 성질 중 하나는, 모든 매개변수(즉, $\boldsymbol{\phi}$ 와 $\boldsymbol{\theta}$)에 대해  
확률적 경사하강법(stochastic gradient descent, SGD)으로 공동 최적화(joint optimization)가 가능하다는 점이다.  

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

그러나 뒤에서 보게 될 것처럼, 좋은 불편 추정량(unbiased estimator)인 $$\tilde{\nabla}_{\boldsymbol{\theta}, \boldsymbol{\phi}} \mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})$$를 사용하면,  
미니배치 SGD도 수행할 수 있다.  

---

비편향(unbiased) ELBO 그래디언트는 생성 모델의 매개변수 $\boldsymbol{\theta}$ 에 대해 비교적 간단하게 계산할 수 있다.

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

> **1) (2.14) → (2.15):**  
>    그래디언트 $\nabla_{\theta}$는 디코더 $p_{\theta}(x,z)$ 의 파라미터에만 의존하므로, 인코더 분포 $q_{\phi}(z\mid x)$와는 무관하다.  
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
>    (2.15)식의 기댓값은 적분 형태로 계산하기 어렵기 때문에, 몬테카를로 추정(Monte Carlo estimation)을 이용해 근사한다.  
>
>    $$
>    \mathbb{E}_{q_{\phi}(z\mid x)}[f(z)]
>    \approx
>    \frac{1}{L}\sum_{l=1}^{L} f(z^{(l)}),
>    \quad z^{(l)} \sim q_{\phi}(z\mid x)
>    $$
>
>    샘플 수 $L$이 충분히 크면 이 근사치는 실제 기댓값에 수렴하며, 이는 대수의 법칙(Law of Large Numbers)에 의해 보장된다.  
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

마지막 줄의 식 (2.17)은 식 (2.15)의 단순한 몬테카를로 추정(Monte Carlo estimator)이다.  
즉, 식 (2.16)과 (2.17)에 등장하는 $\mathbf{z}$는 $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$ 로부터 샘플링된 확률표본이다.  

이 과정을 통해 $\boldsymbol{\theta}$ 에 대한 그래디언트를 근사적으로 계산할 수 있으며,  
이 값은 확률적 경사하강법(SGD)을 이용한 최적화에 직접 사용할 수 있다.
