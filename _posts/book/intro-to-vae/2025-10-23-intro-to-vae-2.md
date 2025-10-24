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

## 2.3 확률적 경사 기반 ELBO 최적화  

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

불편향(unbiased) ELBO 그래디언트는 생성 모델의 매개변수 $\boldsymbol{\theta}$ 에 대해 비교적 간단하게 계산할 수 있다.

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

---

불편향(unbiased) 그래디언트를 변분 파라미터(variational parameters) $\boldsymbol{\phi}$ 에 대해 계산하는 것은  
$\boldsymbol{\theta}$ 에 대한 경우보다 훨씬 더 어렵다.  

이는 ELBO의 기댓값이 분포 $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ 에 대해 정의되어 있는데, 이 분포 자체가 $\boldsymbol{\phi}$ 의 함수이기 때문이다.  

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
>    따라서 인코더의 경우에는 “기댓값 안의 식”과 “확률밀도 함수” 모두에 $\boldsymbol{\phi}$ 가 들어 있어 미분이 훨씬 더 복잡해진다.  

연속적인 잠재변수의 경우, $\nabla_{\boldsymbol{\theta}, \boldsymbol{\phi}} \mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})$의 불편향 추정값(unbiased estimates)을 계산하기 위해  
재매개변수화 기법(reparameterization trick)을 사용할 수 있다.  

이 확률적 추정은 확률적 경사하강법(SGD)을 사용하여 ELBO를 최적화할 수 있게 해 준다.  
알고리즘 1을 참조하라.  

불연속 잠재변수에 대한 변분 방법(variational methods)에 대한 논의는 2.9.1절을 참조하라.

---

**알고리즘 1. ELBO의 확률적 최적화 (Stochastic optimization of the ELBO)**

노이즈는 미니배치 샘플링과 $p(\boldsymbol{\epsilon})$의 샘플링 두 과정 모두에서 발생하기 때문에,  
이 절차는 이중 확률적 최적화(doubly stochastic optimization) 방식이다.  

이 절차는 오토인코딩 변분 베이즈(Auto-Encoding Variational Bayes, AEVB) 알고리즘이라고도 불린다.

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
>      → 각 데이터포인트에 대해 독립적인 노이즈 $\boldsymbol{\epsilon}$ 을 샘플링한다.  
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
ELBO는 변수변환(change of variables)을 통해 $\boldsymbol{\phi}$ 와 $\boldsymbol{\theta}$ 모두에 대해 직접적으로 미분할 수 있다.  

이 방법을 재매개변수화 기법(Reparameterization trick)이라고 하며,  
Kingma & Welling (2014), Rezende et al. (2014) 에 의해 제안되었다.

---

### 2.4.1 변수 변경 (Change of Variables)

먼저, 잠재변수 $\mathbf{z}$ 가 인코더 분포 $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$로부터 샘플링된다고 하자.  

이때 $\mathbf{z}$를 또 다른 확률변수 $\boldsymbol{\epsilon}$의 미분 가능하고 가역적인(invertible) 변환(transformation)으로 표현할 수 있다.

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
그림 3.2는 2차원 예제(2D toy problem)에 대해 결과로 얻어지는 사후분포(posteriors)를 시각화한 것이다.

---

그림 2.3: 재매개변수화 기법(Reparameterization trick)의 예시  

변분 파라미터 $\boldsymbol{\phi}$는 확률변수 $\mathbf{z} \sim q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$를 통해 목적함수 $f$ 에 영향을 미친다.  

우리는 확률적 경사하강법(SGD)을 이용해 목적함수를 최적화하기 위해 $\nabla_{\boldsymbol{\phi}} f$ 를 계산하고자 한다.  

원래 형태(왼쪽)에서는, 확률 변수 $\mathbf{z}$ 를 통해 그래디언트를 직접 역전파(backpropagate)할 수 없기 때문에  
$f$ 를 $\boldsymbol{\phi}$ 에 대해 미분할 수 없다.

우리는 변수 $\mathbf{z}$의 무작위성을,  
$\boldsymbol{\phi}$, $\mathbf{x}$, 그리고 새로 도입된 확률 변수 $\boldsymbol{\epsilon}$의 결정론적이고 미분 가능한 함수로  
변수의 매개변수를 다시 설정(re-parameterizing)함으로써  
‘외부화(externalize)’할 수 있다.

이것은 우리가 ‘$\mathbf{z}$를 통해 역전파(backprop through $\mathbf{z}$)’하고, 그래디언트 $\nabla_{\boldsymbol{\phi}} f$ 를 계산할 수 있게 해준다.

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

그 결과, 단일 데이터 포인트에 대한 ELBO의 단순한 몬테카를로 추정량(estimator) $\tilde{\mathcal{L}}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})$ 을 구성할 수 있다.  

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

이 일련의 연산들은 TensorFlow와 같은 소프트웨어에서 기호적 그래프(symbolic graph) 형태로 표현될 수 있으며,  
파라미터 $\boldsymbol{\theta}$ 와 $\boldsymbol{\phi}$에 대해 손쉽게 미분될 수 있다.  

그 결과로 얻어진 그래디언트 $\nabla_{\boldsymbol{\phi}} \tilde{\mathcal{L}}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})$는  미니배치 SGD를 사용하여 ELBO를 최적화하는 데 사용된다.  
알고리즘 1을 참조하라.  

이 알고리즘은 Kingma와 Welling(2014)에 의해  
처음에는 오토인코딩 변분 베이즈(Auto-Encoding Variational Bayes, AEVB) 알고리즘으로 불렸다.  

보다 일반적으로, 재매개변수화된 ELBO 추정량은  
확률적 그래디언트 변분 베이즈(Stochastic Gradient Variational Bayes, SGVB) 추정량이라고 불린다.  

이 추정량은 또한 모델 파라미터에 대한 사후분포(posterior)를 추정하는 데에도 사용할 수 있으며,  
이에 대한 설명은 Kingma와 Welling(2014)의 부록(Appendix)에 제시되어 있다.

---

#### 불편향성 (Unbiasedness)

이 그래디언트는 정확한 단일 데이터 포인트 ELBO 그래디언트의 불편향 추정량(unbiased estimator)이다.  

즉, 노이즈 $\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})$에 대해 평균을 취하면,  
이 그래디언트는 단일 데이터포인트 ELBO 그래디언트와 동일하다.

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

이 로그-행렬식이 $g(\cdot)$ 와 마찬가지로 $\mathbf{x}$, $\boldsymbol{\epsilon}$, 그리고 $\boldsymbol{\phi}$ 의 함수임을 명확히 하기 위해 $\log d_{\boldsymbol{\phi}}(\mathbf{x}, \boldsymbol{\epsilon})$ 라는 표기를 사용한다.  

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

앞으로 보이겠지만, $\log d_{\boldsymbol{\phi}}(\mathbf{x}, \boldsymbol{\epsilon})$ 를 쉽게 계산할 수 있는 매우 유연한 변환 함수 $g(\cdot)$ 를 구성할 수 있다.  
이를 통해 매우 유연한 추론 모델 $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ 를 얻을 수 있다.

> (1) $p(\boldsymbol{\epsilon})$ 는 우리가 직접 정한 노이즈 분포의 밀도이다  
>     예를 들어 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$ 처럼 선택했기 때문에 그 식을 알고 있다.  
>
> (2) $g(\cdot)$ 는 노이즈를 잠재변수로 바꾸는 함수이다.  
>     $\mathbf{z} = g(\boldsymbol{\epsilon}, \boldsymbol{\phi}, \mathbf{x})$ 로 표현하며,  
>     $g$ 가 가역(invertible)이라면 $\boldsymbol{\epsilon}$ 의 밀도와 $\mathbf{z}$ 의 밀도는  
>     변수 변경 공식(change of variables formula)에 의해 연결된다.  
>
> (3) 식 (2.33)의 뜻  
>     $\log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$는 “노이즈 공간”의 확률을 “잠재변수 공간”의 확률로 바꾼 결과이다.  
>     우리가 알고 있는 것은 $\boldsymbol{\epsilon}$ (노이즈)의 확률분포이기 때문에,  
>     $\mathbf{z} = g(\boldsymbol{\epsilon}, \boldsymbol{\phi}, \mathbf{x})$로 바뀔 때 그 확률이 어떻게 변하는지를 계산해야 한다.  
>     식 (2.33)은 바로 그 변환 관계를 나타낸다.  
>     즉,  
>     - $\log p(\boldsymbol{\epsilon})$: 원래 노이즈의 확률  
>     - $d_{\boldsymbol{\phi}}(\mathbf{x}, \boldsymbol{\epsilon})$: 변환 과정에서 확률이 얼마나 퍼지거나 압축되는지를 나타내는 보정항  
>     이 두 항을 조합하면, 변환된 공간에서의 새로운 확률 $\log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ 를 얻는다.  
>
> (4) 식 (2.34)의 뜻  
>     $d_{\boldsymbol{\phi}}(\mathbf{x}, \boldsymbol{\epsilon})$는 함수 $g$ 가 입력 노이즈 $\boldsymbol{\epsilon}$ 을 출력 $\mathbf{z}$ 로 바꿀 때  
>     “공간을 얼마나 늘이거나 줄이는지”를 나타내는 값이다.  
>     예를 들어,  
>     - 공간이 2배로 늘어나면 확률밀도는 절반으로 줄고,  
>     - 공간이 절반으로 압축되면 확률밀도는 2배로 커진다.  
>     이 변화율을 수학적으로 표현한 것이  
>     야코비안 행렬 $\dfrac{\partial \mathbf{z}}{\partial \boldsymbol{\epsilon}}$ 의 행렬식(det)이다.  
>     로그를 취하면 이 변화를 덧셈 형태로 다룰 수 있어서 계산이 훨씬 단순해진다.  
>
> (5) 왜 야코비안 행렬이 등장하는가  
>     $\mathbf{z}$ 와 $\boldsymbol{\epsilon}$ 은 모두 여러 차원을 가지는 벡터이므로,  
>     각 성분끼리 단순히 1:1 대응하는 것이 아니라, 서로 다른 인덱스들 간에도 영향을 주고받는다.  
>     따라서 한 변수의 변화가 다른 여러 변수에 영향을 미치므로 전체 관계를 편미분으로 표현해야 한다.  
>     이때 사용되는 것이 바로 야코비안 행렬이다.  
>
>$$
>J = \frac{\partial \mathbf{z}}{\partial \boldsymbol{\epsilon}}
>=
>\begin{bmatrix}
>\dfrac{\partial z_1}{\partial \epsilon_1} & \dfrac{\partial z_1}{\partial \epsilon_2} & \cdots & \dfrac{\partial z_1}{\partial \epsilon_n} \\
>\dfrac{\partial z_2}{\partial \epsilon_1} & \dfrac{\partial z_2}{\partial \epsilon_2} & \cdots & \dfrac{\partial z_2}{\partial \epsilon_n} \\
>\vdots & \vdots & \ddots & \vdots \\
>\dfrac{\partial z_m}{\partial \epsilon_1} & \dfrac{\partial z_m}{\partial \epsilon_2} & \cdots & \dfrac{\partial z_m}{\partial \epsilon_n}
>\end{bmatrix}
>$$
>
> 이 행렬의 행렬식(det)은  
> 변환 $g$ 가 공간의 “부피(volume)”를 얼마나 늘리거나 줄였는지를 나타낸다.  
> 예를 들어, $\det(J) > 1$ 이면 공간이 늘어나서 확률밀도가 줄고,  
> $\det(J) < 1$ 이면 공간이 압축되어 확률밀도가 커진다.  
> 따라서 식 (2.34)의 로그 행렬식은  
> 이러한 부피 변화율을 로그 스케일에서 더해 주는 보정항 역할을 한다.

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

여기서 $\mathcal{N}(z_i; \mu_i, \sigma_i^2)$ 는 단변량 가우시안 분포(univariate Gaussian distribution)의 확률밀도함수(PDF)이다.  

> (1) 인코더 신경망의 역할  
>     인코더 $\mathrm{EncoderNeuralNet}_{\boldsymbol{\phi}}(\mathbf{x})$ 는 입력 $\mathbf{x}$ 로부터  
>     잠재분포의 파라미터인 평균 $\boldsymbol{\mu}$ 와 로그표준편차 $\log \boldsymbol{\sigma}$ 를 출력한다.  
>     즉, 인코더는 **데이터 $\mathbf{x}$ 가 주어졌을 때 잠재변수 $\mathbf{z}$ 의 분포를 결정하는 함수**이다.  
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
>     그래서 이 분포를 **분리된 가우시안 분포(factorized Gaussian)** 라고 부른다.  
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
= \sum_i \log \mathcal{N}(\epsilon_i; 0, 1)
- \log \sigma_i
\tag{2.44}
$$

여기서 $\mathbf{z} = g(\boldsymbol{\epsilon}, \boldsymbol{\phi}, \mathbf{x})$ 이다.

>첫 번째 항 $\log p(\boldsymbol{\epsilon})$ 은  
>표준 정규분포 $\mathcal{N}(0, I)$ 의 로그밀도이므로  
>모든 독립 차원에 대한 합으로 쓸 수 있다.  
>
>$$
>\log p(\boldsymbol{\epsilon})
>= \sum_i \log \mathcal{N}(\epsilon_i; 0, 1)
>$$  
>
>두 번째 항은 이미 $\sum_i \log \sigma_i$ 형태로 표현되므로,  
>식 (2.43)에 대입하면  
>
>$$
>\log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})
>= \sum_i \log \mathcal{N}(\epsilon_i; 0, 1)
>- \sum_i \log \sigma_i
>$$  
>
>가 된다.  
>논문에서는 이를 벡터 연산 형태로 간략히 표현하기 위해  
>$\sum$ 기호를 생략하고 다음처럼 표기한다.  
>
>$$
>\log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})
>= \sum_i \log \mathcal{N}(\epsilon_i; 0, 1)
>- \log \sigma_i
>\tag{2.44}
>$$  
>
>실제 의미상으로는 여전히 모든 차원에 대한 합이 내포되어 있다.  

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
> - 대각 원소 $L_{11}, L_{22}, L_{33}$ 는 각 차원의 **분산(scale)** 을 조절하고,  
> - 비대각 원소(off-diagonal) $L_{21}, L_{31}, L_{32}$ 는  
>   차원 간의 **공분산(covariance)** 또는 **상관성(correlation)** 을 반영한다.  
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
>
> (5) 결과적으로  
> 야코비안이 $\mathbf{L}$ 로 주어질 때,  
> 그 로그 행렬식은 위와 같이 대각 원소의 로그합으로 단순화된다.  
> 이는 계산 효율성을 높이고,  
> 공분산을 포함한 복잡한 가우시안 분포에서도  
> 손쉽게 로그 확률을 계산할 수 있도록 해준다.

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
> 결과적으로 $\boldsymbol{\Sigma}=\mathbf{L}\mathbf{L}^{\top}$ 가 되며, 이는 공분산 행렬의 촐레스키 분해(Cholesky decomposition)에 해당한다.

---

원하는 성질, 즉 삼각행렬성과 0이 아닌 대각 원소를 가지는 행렬 $\mathbf{L}$ 을 구성하는 한 가지 방법은 다음과 같다:

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

여기서 $\mathbf{L}_{\text{mask}}$ 는 마스크 행렬로, 대각선 위와 대각선에는 0을, 대각선 아래에는 1을 가지도록 구성된다.  
즉, $\mathbf{L}'$ 의 상삼각 부분은 제거되고 하삼각 부분만 남게 된다.  

> (1) 식 (2.53)의 의미  
> 인코더 신경망은 입력 $\mathbf{x}$ 로부터  
> 평균 $\boldsymbol{\mu}$, 로그표준편차 $\log\boldsymbol{\sigma}$, 그리고 공분산 구조를 학습하기 위한 가중치 행렬 $\mathbf{L}'$ 을 출력한다.  
> 여기서 $\boldsymbol{\sigma}$ 는 각 잠재변수의 단변량 표준편차(대각 항)를, $\mathbf{L}'$ 은 변수 간 상관관계(비대각 항)를 학습한다.  
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

좀 더 일반적으로, $\mathbf{z} = \mathbf{L}\boldsymbol{\epsilon} + \boldsymbol{\mu}$를 (미분 가능하고 비선형적인) 변환들의 연쇄(chain)로 대체할 수 있다.  

이때, 연쇄의 각 단계에서의 야코비안(Jacobian)이 비영(非零) 대각 원소를 가진 삼각행렬(triangular matrix) 형태라면,  
로그 행렬식(log determinant)은 여전히 단순하게 계산된다.  

이 원리는 Inverse Autoregressive Flow (IAF)에서 사용되며,  
이는 *Kingma et al., 2016* 에서 처음 제안되었고 이 책의 3장에서 자세히 논의된다.

---

**알고리즘 2. 단일 데이터포인트 ELBO의 비편향 추정 (Unbiased estimate of single-datapoint ELBO)**  

이 알고리즘은 완전 공분산(full-covariance) 가우시안 추론 모델과  
분리된 베르누이(factorized Bernoulli) 생성 모델을 가지는 예시 VAE의  
단일 데이터포인트(single-datapoint)에 대한 ELBO를 비편향적으로 추정한다.  

여기서 $\mathbf{L}_{\text{mask}}$ 는  
대각선 위와 대각선에 0을, 대각선 아래에는 1을 가지는 마스크 행렬(masking matrix)이다.  

---

**데이터 (Data):**  
- $\mathbf{x}$: 데이터포인트 (필요시 조건부 정보 포함 가능)  
- $\boldsymbol{\epsilon}$: $p(\boldsymbol{\epsilon}) = \mathcal{N}(0, I)$ 로부터의 무작위 샘플  
- $\boldsymbol{\theta}$: 생성 모델(generative model) 파라미터  
- $\boldsymbol{\phi}$: 추론 모델(inference model) 파라미터  
- $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$: 추론 모델 (Inference model)  
- $p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})$: 생성 모델 (Generative model)  

**결과 (Result):**  
- $\tilde{\mathcal{L}}$: 단일 데이터포인트에 대한 ELBO의 비편향 추정치  

---

**절차 (Procedure):**

$$
(\boldsymbol{\mu}, \log \boldsymbol{\sigma}, \mathbf{L}')
\leftarrow \mathrm{EncoderNeuralNet}_{\boldsymbol{\phi}}(\mathbf{x})
\tag{2.53}
$$

$$
\mathbf{L}
\leftarrow \mathbf{L}_{\text{mask}} \odot \mathbf{L}' + \mathrm{diag}(\boldsymbol{\sigma})
\tag{2.54}
$$

$$
\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)
\tag{2.38}
$$

$$
\mathbf{z} = \mathbf{L}\boldsymbol{\epsilon} + \boldsymbol{\mu}
\tag{2.47}
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

> (1) **인코더 단계**  
>    $\mathrm{EncoderNeuralNet}_{\boldsymbol{\phi}}(\mathbf{x})$는 입력 $\mathbf{x}$ 로부터 평균 $\boldsymbol{\mu}$, 로그표준편차 $\log\boldsymbol{\sigma}$, 그리고 초기 공분산 행렬 $\mathbf{L}'$ 을 추정한다.  
>
> (2) **마스크 적용 단계**  
>    마스크 행렬 $\mathbf{L}_{\text{mask}}$ 를 적용해 상삼각 항을 제거하고, 대각에는 $\boldsymbol{\sigma}$를 추가하여  
>   하삼각 형태의 공분산 행렬 $\mathbf{L}$을 생성한다.  
>
> (3) **잠재변수 샘플링**  
>    $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$에 따라 노이즈를 샘플링한 뒤, $\mathbf{z} = \mathbf{L}\boldsymbol{\epsilon} + \boldsymbol{\mu}$ 로 재매개변수화한다.  
>
> (4) **ELBO 항 계산**  
>    - $$\tilde{\mathcal{L}}_{\log q_{\mathbf{z}}}$$: 인코더의 사후 분포 항  
>    - $$\tilde{\mathcal{L}}_{\log p_{\mathbf{z}}}$$: 잠재변수의 사전 분포 항  
>    - $$\tilde{\mathcal{L}}_{\log p_{\mathbf{x}}}$$: 복원된 입력의 로그우도 항  
>
> (5) **ELBO 결합**  
>    최종 ELBO 추정치는 다음과 같다.  
>
>    $$
>    \tilde{\mathcal{L}} = 
>    \tilde{\mathcal{L}}_{\log p_{\mathbf{x}}}
>    + \tilde{\mathcal{L}}_{\log p_{\mathbf{z}}}
>    - \tilde{\mathcal{L}}_{\log q_{\mathbf{z}}}
>    $$  
>
>    이 식은 단일 데이터포인트에 대한 ELBO의 비편향 추정치로 사용된다.