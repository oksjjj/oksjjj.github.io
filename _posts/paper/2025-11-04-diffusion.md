---
layout: post
title: "[논문] Denoising Diffusion Probabilistic Models"
date: 2025-11-04 12:00:00 +0900
categories:
  - "논문"
tags: []
---
> 논문 출처  
> Ho, J., Jain, A., & Abbeel, P.  
> Denoising Diffusion Probabilistic Models.  
> Proceedings of the 34th Conference on Neural Information Processing Systems (NeurIPS), 2020, Vancouver, Canada.  
> <a href="https://arxiv.org/abs/2006.11239" target="_blank">🔗 원문 링크 (arXiv: 2006.11239)</a>  

저자  

- Jonathan Ho  
  UC Berkeley  
  jonathanho@berkeley.edu  

- Ajay Jain  
  UC Berkeley  
  ajayj@berkeley.edu  

- Pieter Abbeel  
  UC Berkeley  
  pabbeel@cs.berkeley.edu

제34회 신경정보처리시스템 학술대회(Neural Information Processing Systems, NeurIPS)  
캐나다 밴쿠버(Vancouver)에서 2020년에 개최.

---

## 초록 (Abstract)  

우리는 확산 확률 모델(diffusion probabilistic models)을 사용하여  
고품질의 이미지 합성 결과를 제시한다.  
이 모델은 비평형 열역학(nonequilibrium thermodynamics)의 고찰에서  
영감을 받은 잠재 변수(latent variable) 모델의 한 종류이다.

우리의 최상의 결과는,  
확산 확률 모델(diffusion probabilistic models)과  
랑주뱅 동역학(Langevin dynamics)을 이용한  
잡음 제거 점수 매칭(denoising score matching) 사이의  
새로운 연결에 따라 설계된 가중 변분 경계(weighted variational bound)에서  
훈련함으로써 얻어진다.  

또한 우리의 모델은  
보통 점진적인 손실 압축 해제(progressive lossy decompression) 방식을  
허용하며, 이는 자기회귀 디코딩(autoregressive decoding)의  
일반화로 해석될 수 있다.  

라벨 정보를 사용하지 않은 CIFAR10 데이터셋(unconditional CIFAR10 dataset)에서  
우리는 Inception 점수 9.46과  
최첨단의 FID 점수(state-of-the-art FID score) 3.17을 얻었다.  

> Inception 점수(Inception Score, IS)는  
> 생성된 이미지가 얼마나 명확하고 다양하게 분포하는지를 평가하는 지표이다.  
> 사전 학습된 Inception 네트워크로 각 이미지의 클래스 확률을 예측하여  
> 각 이미지의 불확실성(개별 엔트로피)이 낮고,  
> 전체 분포의 다양성(전체 엔트로피)은 높을수록 좋은 점수를 얻는다.  
>
> FID(Frechet Inception Distance)는  
> 생성된 이미지 분포와 실제 이미지 분포의 유사도를 평가하는 지표이다.  
> Inception 네트워크의 feature 공간에서  
> 두 분포의 평균과 공분산을 비교하여 계산되며,  
> 값이 낮을수록 생성된 이미지가 실제 데이터 분포에 가깝다는 것을 의미한다.

256×256 LSUN 데이터셋에서는  
ProgressiveGAN과 유사한 샘플 품질을 얻었다.

> ProgressiveGAN(Progressive Growing of GANs)은  
> 2017년에 NVIDIA 연구팀(Karras et al.)이 제안한 생성적 적대 신경망(GAN)의 한 방식이다.  
> 핵심 아이디어는 낮은 해상도에서부터 시작하여 점진적으로 층(layer)을 추가하면서  
> 네트워크를 고해상도로 확장해 가는 것이다.  
>
> 이렇게 하면 학습이 안정적(stable) 이고,  
> 세부 구조가 점진적으로 정교해지며,  
> 매우 고품질의 이미지를 생성할 수 있다.  
>
> ProgressiveGAN은 이후 StyleGAN, StyleGAN2 등의 발전된 모델의  
> 기반이 된 중요한 초기 연구이다.

우리의 구현은 다음에서 확인할 수 있다:  
<a href="https://github.com/hojonathanho/diffusion" target="_blank">https://github.com/hojonathanho/diffusion</a>

---

## 1 서론 (Introduction)

최근 들어 다양한 데이터 유형(data modalities)에서  
모든 종류의 심층 생성 모델(deep generative models)들이  
고품질의 샘플을 보여주고 있다.  

생성적 적대 신경망(Generative Adversarial Networks, GANs),  
자가회귀 모델(autoregressive models),  
흐름 모델(flows), 그리고  
변분 오토인코더(Variational Autoencoders, VAEs)는  
주목할 만한 이미지와 오디오 샘플을 합성해왔다 [14, 27, 3, 58, 38, 25, 10, 32, 44, 57, 26, 33, 45].  

또한 에너지 기반 모델링(energy-based modeling)과  
점수 매칭(score matching)에서도  
GANs가 생성한 이미지에 필적하는 결과를 만들어내는  
놀라운 발전이 있었다 [11, 55].

---

**그림 1:**  
CelebA-HQ 256×256 (왼쪽)과  
라벨이 없는 CIFAR10(unconditional CIFAR10, 오른쪽)에서 생성된 샘플들.

<img src="/assets/img/paper/diffusion/image_1.png" alt="image" width="800px"> 

---

**그림 2:**  
이 연구에서 고려된 유향 그래프 모델(directed graphical model).

<img src="/assets/img/paper/diffusion/image_2.png" alt="image" width="800px"> 

---

이 논문은 확산 확률 모델(diffusion probabilistic models) [53]의 발전을 제시한다.  
확산 확률 모델(간단히 “확산 모델(diffusion model)”이라 부르기로 한다)은  
유한한 시간 후 데이터와 일치하는 샘플을 생성하기 위해  
변분 추론(variational inference)을 사용하여 학습된  
매개변수화된 마르코프 연쇄(parameterized Markov chain)이다.  

이 연쇄의 전이(transition)는 확산 과정을 역전시키도록 학습되는데,  
이 확산 과정은 신호가 파괴될 때까지  
데이터에 점진적으로 잡음을 추가하는 마르코프 연쇄이다.  

확산이 작은 양의 가우시안 잡음으로 구성될 때에는,  
샘플링 연쇄의 전이를 조건부 가우시안(conditional Gaussians)으로 설정하는 것으로 충분하며,  
이로 인해 특히 단순한 신경망 매개변수화(neural network parameterization)가 가능해진다.

---

확산 모델(diffusion models)은 정의하기 쉽고 학습 효율도 높지만,  
우리의 지식으로는 지금까지 그러한 모델이  
고품질 샘플을 생성할 수 있음을 입증한 사례는 없었다.  

우리는 확산 모델이 실제로 고품질 샘플을 생성할 수 있음을 보이며,  
때로는 다른 종류의 생성 모델들에 대한 기존 결과보다  
더 나은 성능을 보이기도 함을 보여준다 (섹션 4).  

또한 우리는 확산 모델의 특정한 매개변수화(parameterization)가  
훈련 중 여러 잡음 수준에서의 잡음 제거 점수 매칭(denoising score matching) 및  
샘플링 중의 어닐링<sup>*</sup>된 랑주뱅 동역학(annealed Langevin dynamics)과
동등함을 드러냄을 보여준다 (섹션 3.2) [55, 61].  

> <sup>*</sup>여기서 "어닐링(annealing)"은 잡음의 세기를 점진적으로 줄여나가며  
> 시스템이 점차 안정된 상태에 수렴하도록 하는 과정을 의미함

우리는 이러한 매개변수화를 사용했을 때  
가장 우수한 샘플 품질 결과를 얻었으며 (섹션 4.2),  
따라서 이 등가성(equivalence)을 본 연구의 주요 기여 중 하나로 간주한다.

---

샘플 품질에도 불구하고,  
우리의 모델은 다른 가능도 기반 모델(likelihood-based models)에 비해  
경쟁력 있는 로그 가능도(log likelihood)를 가지지 않는다.  

(그러나 우리의 모델은  
에너지 기반 모델(energy-based models)과 점수 매칭(score matching)에 대해  
어닐링된 중요도 샘플링(annealed importance sampling)이 산출하는 것으로 보고된,  
가장 큰 추정치보다 더 나은 로그 가능도(log likelihood)를 가진다 [11, 55].)

> 로그 가능도(log likelihood)는  
> 모델이 주어진 데이터를 얼마나 잘 설명하는지를 나타내는 지표이다.  
> 값이 높을수록 모델이 실제 데이터 분포를 더 정확히 근사한다는 뜻이다.  
>  
> 확산 모델은 생성된 이미지의 시각적 품질은 매우 뛰어나지만,  
> 수학적으로 계산되는 확률 기반 평가지표(로그 가능도)는  
> 다른 가능도 기반 모델들보다 낮다는 점을 의미한다.  
>  
> 다만, 여기서 언급된 어닐링된 중요도 샘플링(annealed importance sampling) 은  
> 모델의 로그 가능도를 근사적으로 계산하기 위한 고급 통계 기법으로,  
> 여러 단계에 걸쳐 “온도(temperature)”를 서서히 낮추며  
> 분포 간의 차이를 보정하는 방식이다.  
>  
> 즉, 이 기법으로 추정된 에너지 기반 모델의 가장 높은 추정치와 비교해도,  
> 확산 모델의 로그 가능도가 그보다 더 우수하다는 점을 강조하고 있다.

우리는 우리의 모델의 무손실 부호화 길이(lossless codelength) 대부분이  
지각할 수 없는 이미지 세부 사항들을 설명하는 데 소비된다는 것을 발견했다 (섹션 4.3).  

> 무손실 부호화 길이(lossless codelength)는  
> 데이터를 압축했을 때 정보를 전혀 잃지 않고 표현하는 데 필요한 비트 수를 의미한다.  
>  
> 여기서 “지각할 수 없는 이미지 세부 사항”이란  
> 사람의 눈에는 거의 구분되지 않지만  
> 모델이 데이터의 모든 픽셀 값을 완벽히 맞추기 위해  
> 불필요하게 많은 정보를 사용하고 있다는 뜻이다.  
>  
> 즉, 확산 모델이 실제로는 시각적으로 중요하지 않은 미세한 잡음 수준의 정보까지  
> 부호화하려고 하기 때문에,  
> 전체 부호화 길이의 대부분이 이런 사소한 세부 묘사에 소모된다는 의미이다.

우리는 이 현상에 대해  
손실 압축(lossy compression)의 언어로 보다 정제된 분석을 제시하며,  
확산 모델의 샘플링 절차가  
일반적인 자기회귀 모델(autoregressive models)에서 일반적으로 가능한 것보다  
훨씬 더 일반화된 비트 순서(bit ordering)를 따라 진행되는  
자기회귀 디코딩(autoregressive decoding)과 유사한  
일종의 점진적 디코딩(progressive decoding)임을 보인다.

> 이 문장은 다음과 같은 의미를 담고 있다.  
>  
> 먼저, 저자들은 모델이 불필요하게 많은 정보를 부호화하는 현상을  
> 손실 압축(lossy compression), 즉 덜 중요한 정보를 버리고 핵심만 남기는 방식으로  
> 다시 해석하고 있다는 뜻이다.  
>  
> 그리고 확산 모델의 샘플링 과정이  
> 일반적인 자기회귀 모델(autoregressive model)처럼  
> 한 단계씩 점진적으로 데이터를 복원하는 “디코딩 과정”과 유사하지만,  
> 그보다 훨씬 유연하고 일반화된 순서(bit ordering) 로 진행된다고 설명한다.  
>  
> 즉, 확산 모델은 자기회귀 디코딩보다  
> 더 세밀하고 연속적인 방식으로 데이터를 재구성하는  
> 점진적 복원(decoding) 과정으로 볼 수 있다는 의미이다.

---

## 2 배경 (Background)

확산 모델(diffusion models) [53]은  
다음 형태의 잠재 변수(latent variable) 모델이다:  

$$ p_\theta(\mathbf{x}_0) := \int p_\theta(\mathbf{x}_{0:T}) \, d\mathbf{x}_{1:T}, $$  

여기서 $\mathbf{x}_1, \dots, \mathbf{x}_T$ 는  
데이터 $\mathbf{x}_0 \sim q(\mathbf{x}_0)$ 와 동일한 차원을 가지는 잠재 변수(latents)이다.  

> 이 수식은 확산 모델이 잠재 변수 모델(latent variable model) 로 구성된다는 것을 의미한다.  
>  
> 즉, 데이터 $\mathbf{x}_0$ 는 단일 확률 변수로 직접 생성되는 것이 아니라,  
> 여러 단계의 잠재 변수 $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T$ 를 거쳐  
> 점진적으로 생성된다는 뜻이다.  
>  
> $$p_\theta(\mathbf{x}_0)$$ 는 최종적으로 우리가 얻고자 하는 데이터의 확률 분포이며,  
> 이 분포는 전체 잠재 변수들의 결합 분포 $$p_\theta(\mathbf{x}_{0:T})$$ 를  
> $$\mathbf{x}_1$$ 부터 $$\mathbf{x}_T$$ 까지 적분(marginalization)하여 얻는다.  
>  
> 다시 말해, $$\mathbf{x}_1$$~$$\mathbf{x}_T$$ 는  
> 데이터 생성 과정 중간의 “숨겨진 단계(hidden steps)”들이며,  
> 모델은 이 잠재 변수들을 통해 데이터의 복잡한 분포를 학습한다.

결합 분포(joint distribution) $p_\theta(\mathbf{x}_{0:T})$ 는  
역과정(reverse process)이라고 불리며,  
이는 학습된 가우시안 전이(Gaussian transitions)를 갖는  
마르코프 연쇄(Markov chain)로 정의된다.  

> 결합 분포 $p_\theta(\mathbf{x}_{0:T})$ 는  
> 데이터 $\mathbf{x}_0$ 부터 잠재 변수 $\mathbf{x}_T$ 까지의  
> 전체 생성 경로 전체를 아우르는 확률 분포를 의미한다.  
>  
> 이를 “역과정(reverse process)”이라고 부르는 이유는,  
> 실제 확산 과정이 데이터를 점점 노이즈로 변환시키는 방향으로 진행되는 반면,  
> 모델은 그 반대 방향으로, 즉 노이즈로부터 데이터를 복원하는 방향으로  
> 학습되기 때문이다.  
>  
> 또한 이 역과정은 마르코프 연쇄(Markov chain) 형태로 정의되는데,  
> 이는 각 단계의 상태 $\mathbf{x}_{t-1}$ 이  
> 바로 이전 단계의 상태 $\mathbf{x}_t$ 에만 의존한다는 뜻이다.  
>  
> 이때 각 전이 $$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$ 는  
> 평균 $\mu_\theta$ 와 공분산 $\Sigma_\theta$ 를 가지는  
> 가우시안 분포(Gaussian transition) 로 모델링되어,  
> 노이즈 제거 과정을 확률적으로 표현한다.

이 연쇄는 다음에서 시작한다:  
$p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; \mathbf{0}, \mathbf{I})$  

그리고 다음과 같이 표현된다:  

$$
p_\theta(\mathbf{x}_{0:T}) := p(\mathbf{x}_T) \prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t), 
\quad p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) := 
\mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))
\tag{1}
$$

> 이 식은 확산 모델의 생성 과정(generative process) 을 수학적으로 정의한 것이다.  
>  
> 먼저, $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; \mathbf{0}, \mathbf{I})$ 는  
> 가장 마지막 단계의 잠재 변수 $\mathbf{x}_T$ 가  
> 평균이 0, 공분산이 단위행렬인 표준 가우시안 분포(standard normal distribution) 로부터  
> 샘플링된다는 뜻이다.  
>  
> 이후 각 단계는 바로 이전 단계의 변수 $$\mathbf{x}_{t-1}$$ 을  
> 현재 변수 $$\mathbf{x}_t$$ 로부터 확률적으로 복원하는 과정을 거치며,  
> 이 전이는 $$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$ 로 표현된다.  
>  
> 특히 이 전이는  
> 평균이 $$\mu_\theta(\mathbf{x}_t, t)$$,  
> 공분산이 $$\Sigma_\theta(\mathbf{x}_t, t)$$ 인  
> 가우시안 분포 $$\mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))$$ 로  
> 정의된다.  
>  
> 이는 모델이 각 단계에서 노이즈를 제거하며  
> 다음 상태를 확률적으로 예측하는 것을 의미한다.  
>  
> 전체 생성 과정은 마르코프 성질에 따라  
> 모든 단계의 전이 확률들을 곱한 형태로 나타낼 수 있으며,  
> 이를 통해 최종적으로 데이터 $\mathbf{x}_0$ 까지 점진적으로 복원된다.  
>  
> 즉, 모델은  
> 노이즈에서 시작해($$\mathbf{x}_T$$),  
> 점차 노이즈를 제거하며($$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$),  
> 최종적으로 데이터 $$\mathbf{x}_0$$ 를 생성하는  
> 확률적 복원 절차(stochastic denoising process) 를 수행한다.

---
확산 모델(diffusion models)을  
다른 유형의 잠재 변수 모델(latent variable models)과 구별하는 것은,  
순방향 과정(forward process) 또는 확산 과정(diffusion process) 이라고 불리는  
근사 사후분포(approximate posterior) $q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)$ 가  
마르코프 연쇄(Markov chain)로 고정되어 있으며,  
이는 분산 스케줄(variance schedule) $\beta_1, \dots, \beta_T$ 에 따라  
데이터에 점진적으로 가우시안 잡음을 추가한다는 점이다.

$$
q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) := \prod_{t=1}^{T} q(\mathbf{x}_t \mid \mathbf{x}_{t-1}), 
\quad q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) := 
\mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \, \mathbf{x}_{t-1}, \, \beta_t \mathbf{I})
\tag{2}
$$

> $$\mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$$ 는  
> 이전 상태 $$\mathbf{x}_{t-1}$$ 의 일부를 남기고, 나머지를 잡음으로 채워  
> $$\mathbf{x}_t$$ 를 만드는 과정을 나타낸다.  
>  
> $(1 - \beta_t)$ 는 신호를 유지하는 비율,  
> $\beta_t$ 는 잡음을 추가하는 비율을 의미한다.  
>  
> 제곱근(√)은 분산이 일정하게 유지되도록 하기 위한 것이다.  
> 즉, 신호와 잡음의 분산이 $(1 - \beta_t) + \beta_t = 1$ 이 되게 하여  
> 값이 발산하지 않고 안정적으로 확산되도록 한다.  
>  
> 결과적으로 $\sqrt{1 - \beta_t}$ 는 “신호 유지 세기”,  
> $\sqrt{\beta_t}$ 는 “잡음 세기”를 조절하는 계수이다.

---

학습은 음의 로그 가능도(negative log likelihood)에 대한  
통상적인 변분 경계(variational bound)를 최적화함으로써 수행된다:

$$
\mathbb{E}[-\log p_\theta(\mathbf{x}_0)] 
\le 
\mathbb{E}_q \left[
-\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}
\right]
=
\mathbb{E}_q \left[
-\log p(\mathbf{x}_T)
-
\sum_{t \ge 1}
\log
\frac{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}{q(\mathbf{x}_t \mid \mathbf{x}_{t-1})}
\right]
=: \mathcal{L}
\tag{3}
$$

> 1. 데이터의 주변 확률(marginal likelihood) 은 다음과 같이 정의된다.  
>
>    $$
>    p_\theta(\mathbf{x}_0) = \int p_\theta(\mathbf{x}_{0:T}) \, d\mathbf{x}_{1:T}
>    $$
>    
>    이는 모든 잠재 변수 $\mathbf{x}_1, \dots, \mathbf{x}_T$ 를 적분해  
>    데이터 $\mathbf{x}_0$ 의 확률을 계산한 것이다.  
>
> 2. 이 적분은 계산이 어렵기 때문에,  
>    계산 가능한 근사 분포 $q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)$ 를 곱하고 나눈다.  
>
>    $$
>    \log p_\theta(\mathbf{x}_0)
>    = \log \int q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)
>    \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \, d\mathbf{x}_{1:T}
>    $$
>
>    이렇게 하면 $q$ 를 기대값(expectation) 형태로 표현할 수 있다.  
>
> 3. 젠센 부등식(Jensen’s inequality) 을 적용하기 전에,  
>    먼저 그 정의를 간단히 살펴보면 다음과 같다.  
>
>    - 함수 $f(x)$ 가 오목(concave) 일 때  
>      확률 변수 $X$ 에 대해 다음이 성립한다:
>
>      $$
>      f(\mathbb{E}[X]) \ge \mathbb{E}[f(X)]
>      $$
>
>    - 로그 함수 $\log(x)$ 는 오목 함수이므로 다음이 성립한다:
>
>      $$
>      \log \mathbb{E}[X] \ge \mathbb{E}[\log X]
>      $$
>
>    이 성질을 위 식에 적용하면,  
>    로그 안쪽의 적분(기댓값)을 바깥으로 이동시킬 수 있다.
>
>    따라서 다음의 부등식이 성립한다:
>
>    $$
>    \log p_\theta(\mathbf{x}_0)
>    = \log \mathbb{E}_q \left[\frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}\right]
>    \ge 
>    \mathbb{E}_q \left[\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}\right]
>    $$
>
>    음의 로그를 취하면 부호가 반대가 되어  
>    식 (3)의 변분 하한(variational lower bound) 형태가 얻어진다.  
>
> 4. 확산 모델은  
>
>    $$p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T)\prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t), \quad q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$$  
>
>    로 정의되므로, 로그 비율은 다음처럼 전개된다.  
>
>    $$
>    -\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}
>    = -\log p(\mathbf{x}_T)
>    - \sum_{t \ge 1} \log \frac{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}{q(\mathbf{x}_t \mid \mathbf{x}_{t-1})}
>    $$
>
> 5. 이를 $q$ 에 대한 기댓값으로 취하면 다음과 같다.  
>
>    $$
>    \mathbb{E}[-\log p_\theta(\mathbf{x}_0)]
>    \le
>    \mathbb{E}_q \left[
>    -\log p(\mathbf{x}_T)
>    - \sum_{t \ge 1} \log
>    \frac{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}{q(\mathbf{x}_t \mid \mathbf{x}_{t-1})}
>    \right]
>    =: \mathcal{L}
>    $$

순방향 과정(forward process)의 분산(variances) $\beta_t$ 는  
재매개변수화(reparameterization) [33]를 통해 학습되거나,  
혹은 하이퍼파라미터(hyperparameters)로 고정될 수 있다.  

역과정(reverse process)의 표현력(expressiveness)은  
$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ 에서  
가우시안 조건부(Gaussian conditionals)를 선택함으로써  
부분적으로 보장된다.  

이는 $\beta_t$ 가 작을 때  
순방향 과정과 역방향 과정이  
같은 함수적 형태(functional form)를 가지기 때문이다 [53].

> 순방향 과정의 분산 $\beta_t$ 는  
> 각 단계에서 얼마나 강한 잡음을 추가할지를 결정하는 값이다.  
>  
> 이 값은 모델이 직접 학습하도록 설정할 수도 있고  
> 사람이 미리 고정된 스케줄(hyperparameter)로 지정할 수도 있다.  
> 예를 들어, $\beta_t$ 를 선형적으로 혹은 지수적으로 증가시키는 스케줄이 자주 사용된다.  
>  
> 역과정 $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ 은  
> 노이즈가 추가된 데이터를 다시 복원하는 확률 분포인데,  
> 이를 가우시안 분포 형태로 설정하면  
> 순방향 과정과 동일한 수학적 구조를 갖게 된다.  
>  
> 특히 $\beta_t$ 가 작을 경우,  
> 순방향 과정의 가우시안 전이와  
> 역과정의 가우시안 복원이 거의 같은 형태를 가지므로,  
> 모델이 학습해야 할 관계가 단순해지고  
> 복원(샘플링) 과정의 표현력이 안정적으로 보장된다.

순방향 과정의 주목할 만한 성질 중 하나는,  
임의의 시점 $t$ 에서 폐형식(closed form)으로  
$\mathbf{x}_t$ 를 샘플링할 수 있다는 점이다.  

$\alpha_t := 1 - \beta_t$ 및  
$$\bar{\alpha}_t := \prod_{s=1}^{t} \alpha_s$$ 라는 표기를 사용하면,  
다음과 같은 식이 성립한다:

$$
q(\mathbf{x}_t \mid \mathbf{x}_0)
=
\mathcal{N}(\mathbf{x}_t;
\sqrt{\bar{\alpha}_t}\mathbf{x}_0,
(1 - \bar{\alpha}_t)\mathbf{I})
\tag{4}
$$

> 우선 다음과 같이 정의한다:
>
> $$
> \alpha_t = 1 - \beta_t, 
> \quad 
> \bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i
> $$
>
> 순방향 과정은 다음과 같은 가우시안 형태로 표현된다:
>
> $$
> \mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1},
> \quad \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(0, \mathbf{I})
> $$
>
> 이 식을 반복적으로 전개하면 다음과 같다.
>
> 첫 번째 단계:
>
> $$
> \mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}
> $$
>
> 두 번째 단계:
>
> $$
> \mathbf{x}_{t-1} = \sqrt{\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{1 - \alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}
> $$
>
> 이를 대입하면,
>
> $$
> \mathbf{x}_t 
> = \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2}
> + \sqrt{\alpha_t(1 - \alpha_{t-1})}\boldsymbol{\epsilon}_{t-2}
> + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}
> $$
>
> “서로 다른 분산을 가진 두 가우시안 분포를 결합할 때”  
> 새로운 분산이 두 분산의 합으로 표현된다.
>
> 예를 들어,  
> $\mathcal{N}(0, \sigma_1^2\mathbf{I})$ 와 $\mathcal{N}(0, \sigma_2^2\mathbf{I})$ 를 결합하면  
> 결과는 $\mathcal{N}(0, (\sigma_1^2 + \sigma_2^2)\mathbf{I})$ 가 된다.  
>
> 이 원리를 적용하면,  
> 단계별 잡음이 합쳐질 때 표준편차는 다음과 같이 단순화된다:
>
> $$
> \sqrt{(1 - \alpha_t) + \alpha_t(1 - \alpha_{t-1})}
> = \sqrt{1 - \alpha_t \alpha_{t-1}}
> $$
>
> ---
>
> 위 과정을 계속 반복하면, 일반적으로 다음 형태로 수렴한다:
>
> $$
> \mathbf{x}_t
> = \sqrt{\bar{\alpha}_t}\mathbf{x}_0
> + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon},
> \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})
> $$
>
> ---
>
> 위 식은 $\mathbf{x}_t$ 가 원본 데이터 $\mathbf{x}_0$ 와  
> 단일 가우시안 잡음 $\boldsymbol{\epsilon}$ 의 선형 결합(linear combination)으로 표현된다는 것을 의미한다.  
>  
> 따라서 $\mathbf{x}_t$ 는 확률 변수 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$ 를 따르며,  
> 평균(mean)은 $\sqrt{\bar{\alpha}_t}\mathbf{x}_0$,  
> 분산(covariance)은 $(1 - \bar{\alpha}_t)\mathbf{I}$ 인 가우시안 분포가 된다.  
>  
> 즉, 이 관계를 확률 분포 형태로 쓰면 다음과 같으며,  
> 이것이 바로 식 (4)이다:
>
> $$
> q(\mathbf{x}_t \mid \mathbf{x}_0)
> =
> \mathcal{N}\!\left(
> \mathbf{x}_t;
> \sqrt{\bar{\alpha}_t}\mathbf{x}_0,
> (1 - \bar{\alpha}_t)\mathbf{I}
> \right)
> \tag{4}
> $$

따라서 확률적 경사 하강법(stochastic gradient descent)을 사용하여  
$\mathcal{L}$ 의 임의 항(random terms)을 최적화함으로써  
효율적인 학습이 가능하다.

추가적인 향상은  
식 (3)의 $\mathcal{L}$ 을 다음과 같이 다시 씀으로써  
분산 감소(variance reduction)로부터 얻어진다:

$$
\mathbb{E}_q \biggl[
\underbrace{
D_{\mathrm{KL}}\!\left(q(\mathbf{x}_T \mid \mathbf{x}_0) \parallel p(\mathbf{x}_T)\right)
}_{L_T}
+ 
\sum_{t > 1}
\underbrace{
D_{\mathrm{KL}}\!\left(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) 
\parallel p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)\right)
}_{L_{t-1}}
- 
\underbrace{
\log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)
}_{L_0}
\biggr]
\tag{5}
$$

(자세한 내용은 부록 A(Appendix A) 를 참고하라.)  

> **식 (5)의 도출 과정**  
>
> Appendix A에서는 식 (3)에서 제시된 변분 하한(variational lower bound)을  
> 보다 분산이 낮은(reduced variance) 형태로 다시 전개하여  
> 식 (5)를 얻는 과정을 단계별로 설명한다.  
>
> ---
>
> **1. 기본 형태로부터 시작 (식 17)**  
>
> 변분 하한 $\mathcal{L}$ 은 다음과 같이 정의된다.
>
> $$
> \mathcal{L}
> = \mathbb{E}_q \left[
> -\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}
> \right]
> \tag{17}
> $$
>
> 이는 전체 데이터 생성 확률과 근사 분포 간의 로그 비율의 기댓값으로,  
> 변분 하한(VLB, variational lower bound)의 기본 형태를 나타낸다.
>
> ---
>
> **2. 공통 항 분리 (식 18)**  
>
> 결합 확률 분포 $$p_\theta(\mathbf{x}_{0:T})$$ 와 $$q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)$$ 는  
> 모두 마르코프 연쇄(Markov chain) 구조를 가지므로,  
> 각 조건부 확률의 곱 형태로 전개할 수 있다:
>
> $$
> p_\theta(\mathbf{x}_{0:T})
> = p(\mathbf{x}_T) \prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t),
> \quad
> q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)
> = \prod_{t=1}^{T} q(\mathbf{x}_t \mid \mathbf{x}_{t-1})
> $$
>
> 이를 식 (17)에 대입하면 다음과 같이 된다.
>
> $$
> \mathcal{L}
> = \mathbb{E}_q \left[
> -\log p(\mathbf{x}_T)
> - \sum_{t \ge 1}
> \log
> \frac{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}{q(\mathbf{x}_t \mid \mathbf{x}_{t-1})}
> \right]
> \tag{18}
> $$
>
> 첫 번째 항은 종단 노이즈 분포 $p(\mathbf{x}_T)$ 의 로그 우도를,  
> 두 번째 항은 단계별 전이 확률의 로그 비율을 나타낸다.
>
> ---
>
> **3. 첫 번째 항과 마지막 항의 분리 (식 19)**  
>
> $t = 1$ 항을 별도로 분리하여,  
> 마지막 복원 항 $p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)$ 을 분리해낸다.
>
> $$
> \mathcal{L}
> = \mathbb{E}_q \left[
> -\log p(\mathbf{x}_T)
> - \sum_{t > 1} \log
> \frac{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}{q(\mathbf{x}_t \mid \mathbf{x}_{t-1})}
> - \log
> \frac{p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)}{q(\mathbf{x}_1 \mid \mathbf{x}_0)}
> \right]
> \tag{19}
> $$
>
> 여기서 마지막 항은 데이터 복원 단계(reconstruction term)를,  
> 나머지 합은 중간 단계의 확률 전이(term transition)를 의미한다.
>
> ---
>
> **4. 조건부 확률의 재정렬 (식 20–21 유도)**  
>
> 순방향 과정의 확률 $$q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$$ 은  
> 사후 확률 형태 $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ 로 변환될 수 있다.  
> 결합 확률의 성질로부터 다음이 성립한다:
>
> $$
> q(\mathbf{x}_{t-1}, \mathbf{x}_t \mid \mathbf{x}_0)
> = q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)\, q(\mathbf{x}_t \mid \mathbf{x}_0)
> = q(\mathbf{x}_t \mid \mathbf{x}_{t-1})\, q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)
> $$
>
> 따라서 다음 관계식을 얻는다.
>
> $$
> q(\mathbf{x}_t \mid \mathbf{x}_{t-1})
> = \frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)\, q(\mathbf{x}_t \mid \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}
> \tag{A}
> $$
>
> 식 (A)를 이용하면 분모의 $$q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$$ 를 다음과 같이 바꿀 수 있다:
>
> $$
> \frac{1}{q(\mathbf{x}_t \mid \mathbf{x}_{t-1})}
> =  \frac{1}{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)}\,
> \frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}{q(\mathbf{x}_t \mid \mathbf{x}_0)}
> $$
>
> 이를 식 (19)에 대입하면 다음이 된다.
>
> $$
> \mathcal{L}
> = \mathbb{E}_q \left[
> -\log p(\mathbf{x}_T)
> - \sum_{t > 1}
> \log \frac{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)}
> \cdot
> \frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}{q(\mathbf{x}_t \mid \mathbf{x}_0)}
> - \log \frac{p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)}{q(\mathbf{x}_1 \mid \mathbf{x}_0)}
> \right]
> \tag{20}
> $$
>
> 여기서 두 번째 로그항은  
> $t$ 에 대해 망원급수(telescoping sum) 형태로 대부분 상쇄되어,
>
> $$
> \sum_{t>1} \log \frac{q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}{q(\mathbf{x}_t \mid \mathbf{x}_0)}
> = \log q(\mathbf{x}_1 \mid \mathbf{x}_0) - \log q(\mathbf{x}_T \mid \mathbf{x}_0)
> $$
>
> 가 된다.  
>
> ---
>
> **이후 망원항을 정리하면 식 (21)로 단순화된다.**
>
> $$
> \mathcal{L}
> = \mathbb{E}_q \left[
> -\log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T \mid \mathbf{x}_0)}
> - \sum_{t > 1}
> \log \frac{p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)}{q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)}
> - \log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)
> \right]
> \tag{21}
> $$
>
> 위 식에서  
>
> $$-\log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T \mid \mathbf{x}_0)}$$ 
> 
> 항은  
> KL 발산 $$D_{\mathrm{KL}}(q(\mathbf{x}_T \mid \mathbf{x}_0) \parallel p(\mathbf{x}_T))$$ 의 형태로 변환되며,  
> 이후 식 (22)로 이어진다.
>
> ---
>
> **5. KL 발산 형태로 표현 (식 22)**  
>
> 위 식 (20)은 KL 발산의 정의를 이용하여 다음과 같이 정리된다.
>
> $$
> \mathcal{L}
> = \mathbb{E}_q \left[
> D_{\mathrm{KL}}(q(\mathbf{x}_T \mid \mathbf{x}_0) \parallel p(\mathbf{x}_T))
> + \sum_{t > 1}
> D_{\mathrm{KL}}(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)
> \parallel p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t))
> - \log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)
> \right]
> \tag{22}
> $$
>
> 이 식이 본문에서 제시된 **식 (5)** 와 동일하다.  
> 즉, 변분 하한을 KL 발산 항들의 합으로 표현한 결과이다.  
> 
> ---
> 
> **6. 왜 “분산이 낮은(reduced variance)” 형태인가**  
> 
> 식 (5)의 각 KL 항은  
> 확률적 샘플링 대신 폐형식(closed-form) 으로 계산 가능한 가우시안 간 KL로 구성된다.  
> 기존 ELBO(식 3)는  
> 단일 샘플 $$\mathbf{x}_0 \sim q(\mathbf{x}_0)$$ 을 사용하여  
> $$\log p_\theta(\mathbf{x}_0)$$ 를 직접 추정해야 하므로  
> 몬테카를로 추정에 의한 분산이 크다.  
> 
> 반면 식 (5)는  
> - $q(\mathbf{x}_T \mid \mathbf{x}_0)$,  
> - $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$,  
> - $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$  
> 모두 가우시안 분포로 명시적 파라미터화가 되어 있고,  
> 각 KL 항이 닫힌 형태로 계산되므로,  
> 표본 추출에 의한 분산이 발생하지 않는다.  
> 
> 즉,  
> 식 (5)는 원래의 ELBO를  
> “샘플링 기반 추정(Monte Carlo Estimation)”에서  
> “가우시안 KL 기반의 결정적 계산(Deterministic KL Computation)”으로 변환한 형태이며,  
> 따라서 훨씬 분산이 낮고 안정적인 학습 목표를 제공한다.  

식의 각 항에 붙은 레이블(label)은 3절(Section 3)에서 사용된다.  
식 (5)는 KL 발산(KL divergence)을 사용하여  
$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$ 과  
순방향 과정의 사후분포(forward process posterior)  
$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$$ 를 직접 비교한다.  
이 비교는 $\mathbf{x}_0$ 가 주어졌을 때 계산이 용이하다(tractable).

$$
q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)
=
\mathcal{N}\!\left(\mathbf{x}_{t-1};
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0),
\tilde{\beta}_t \mathbf{I}\right),
\tag{6}
$$

여기서  

$$
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0)
:=
\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}\mathbf{x}_0
+ \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\mathbf{x}_t,
\quad
\tilde{\beta}_t := \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\beta_t
\tag{7}
$$

> **식 (6)의 도출 과정**  
>
> 식 (6)은 순방향 과정  
> $$q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$$  
> 과 초기 상태 $$\mathbf{x}_0$$ 의 결합 분포로부터  
> 사후분포 $$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$$  
> 를 유도하는 과정이다.  
>
> ---
>
> **1. 순방향 과정의 정의**  
>
> 순방향 확산 과정은 다음의 가우시안 형태로 정의된다.  
>
> $$
> q(\mathbf{x}_t \mid \mathbf{x}_{t-1})
> = \mathcal{N}\!\left(
> \mathbf{x}_t;
> \sqrt{\alpha_t}\mathbf{x}_{t-1},
> (1 - \alpha_t)\mathbf{I}
> \right)
> $$
>
> 이 과정을 $t$ 스텝까지 누적하면,  
> 초기 상태 $\mathbf{x}_0$ 로부터의 직접적인 전이 확률은  
> 다음과 같다.  
>
> $$
> q(\mathbf{x}_t \mid \mathbf{x}_0)
> = \mathcal{N}\!\left(
> \mathbf{x}_t;
> \sqrt{\bar{\alpha}_t}\mathbf{x}_0,
> (1 - \bar{\alpha}_t)\mathbf{I}
> \right),
> \quad
> \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s
> $$
>
> ---
>
> **2. 사후분포의 정의**  
>
> 우리가 구하고자 하는 것은  
> $$\mathbf{x}_t$$ 와 $$\mathbf{x}_0$$ 가 주어졌을 때의  
> $$\mathbf{x}_{t-1}$$ 의 조건부 분포이다.  
>
> 결합 확률의 정의로부터 다음이 성립한다.  
>
> $$
> q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)
> = \frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0)\, q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}{q(\mathbf{x}_t \mid \mathbf{x}_0)}
> $$
>
> 여기서 마르코프 성질(Markov property), 즉  
> “현재 상태는 바로 이전 상태에만 의존하고  
> 그보다 앞선 상태에는 직접 의존하지 않는다”는 성질에 의해  
>
> $$
> q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0)
> = q(\mathbf{x}_t \mid \mathbf{x}_{t-1})
> $$
>
> 이므로 식은 다음과 같이 단순화된다.  
>
> $$
> q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)
> = \frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1})\, q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}{q(\mathbf{x}_t \mid \mathbf{x}_0)}
> $$
>
> 마지막 항 $$q(\mathbf{x}_t \mid \mathbf{x}_0)$$ 은  
> $$\mathbf{x}_{t-1}$$ 과 무관한 정규화 상수이므로,  
> 비례식으로 간단히 쓸 수 있다.  
>
> $$
> q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)
> \propto q(\mathbf{x}_t \mid \mathbf{x}_{t-1})\, q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)
> $$
>
> ---
>
> **3. 가우시안 곱의 일반형 정리**  
>
> 두 가우시안 $\mathcal{N}(\mathbf{x}; \mu_1, \Sigma_1)$ 과  
> $\mathcal{N}(\mathbf{x}; \mu_2, \Sigma_2)$ 의 곱은  
> 또 다른 가우시안 형태로 표현된다.  
>
> $$
> \mathcal{N}(\mathbf{x}; \tilde{\mu}, \tilde{\Sigma}),
> \quad
> \tilde{\Sigma} = (\Sigma_1^{-1} + \Sigma_2^{-1})^{-1},
> \quad
> \tilde{\mu} = \tilde{\Sigma}(\Sigma_1^{-1}\mu_1 + \Sigma_2^{-1}\mu_2)
> $$
>
> ---
>
> **4. 각 항의 구체적 대입**  
>
> 첫 번째 항 $$q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$$ 는  
> 순방향 과정에서 다음의 가우시안 형태로 정의된다.  
>
> $$
> q(\mathbf{x}_t \mid \mathbf{x}_{t-1})
> = \mathcal{N}\!\left(
> \mathbf{x}_t;
> \sqrt{\alpha_t}\mathbf{x}_{t-1},
> (1 - \alpha_t)\mathbf{I}
> \right)
> $$
>
> 그러나 우리는 $$\mathbf{x}_{t-1}$$ 에 대한 사후분포  
> $$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$$ 를 구해야 하므로,  
> 위 확률을 $\mathbf{x}_{t-1}$을 변수로 한 형태로 다시 써야 한다.  
>
> 이를 위해 다음의 확률 밀도 함수를 생각한다.  
>
> $$
> q(\mathbf{x}_t \mid \mathbf{x}_{t-1})
> \propto
> \exp\!\left(
> -\frac{1}{2(1 - \alpha_t)}
> \|\mathbf{x}_t - \sqrt{\alpha_t}\mathbf{x}_{t-1}\|^2
> \right)
> $$
>
> 제곱항을 $\mathbf{x}_{t-1}$ 에 대해 전개하면,
>
> $$
> \|\mathbf{x}_t - \sqrt{\alpha_t}\mathbf{x}_{t-1}\|^2
> = \alpha_t\|\mathbf{x}_{t-1}\|^2 - 2\sqrt{\alpha_t}\mathbf{x}_t^\top \mathbf{x}_{t-1} + \|\mathbf{x}_t\|^2
> $$
>
> 상수항 $$\|\mathbf{x}_t\|^2$$ 는 $$\mathbf{x}_{t-1}$$ 에 의존하지 않으므로 무시할 수 있다.  
> 남은 부분을 완전제곱(completing the square) 형태로 정리하면,
>
> $$
> q(\mathbf{x}_t \mid \mathbf{x}_{t-1})
> \propto
> \exp\!\left(
> -\frac{\alpha_t}{2(1 - \alpha_t)}
> \left\|
> \mathbf{x}_{t-1} - \frac{\mathbf{x}_t}{\sqrt{\alpha_t}}
> \right\|^2
> \right)
> $$
>
> 따라서 이는 $\mathbf{x}_{t-1}$ 에 대한 가우시안으로 다음과 같이 다시 쓸 수 있다.
>
> $$
> q(\mathbf{x}_t \mid \mathbf{x}_{t-1})
> = \mathcal{N}\!\left(
> \mathbf{x}_{t-1};
> \frac{\mathbf{x}_t}{\sqrt{\alpha_t}},
> \frac{1 - \alpha_t}{\alpha_t}\mathbf{I}
> \right)
> $$
>
> 확산 계수 $\beta_t = 1 - \alpha_t$ 를 이용하면,  
> 첫 번째 항 $$q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$$의 평균과 분산은 다음과 같이 정리된다.
>
> $$
> \mu_1 = \frac{\mathbf{x}_t}{\sqrt{\alpha_t}}, \quad
> \Sigma_1 = \frac{\beta_t}{\alpha_t}\mathbf{I}
> $$
>
> 두 번째 항 $q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)$ 은  
>
> $$
> \mu_2 = \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0, \quad
> \Sigma_2 = (1 - \bar{\alpha}_{t-1})\mathbf{I}
> $$
>
> ---
> **5. 평균과 분산의 도출 (식 7)**  
>
> 앞서 얻은 두 항을 다시 상기하자.  
>
> 첫 번째 항:  
>
> $$
> q(\mathbf{x}_t \mid \mathbf{x}_{t-1})
> = \mathcal{N}\!\left(
> \mathbf{x}_{t-1};
> \mu_1 = \frac{\mathbf{x}_t}{\sqrt{\alpha_t}},
> \Sigma_1 = \frac{\beta_t}{\alpha_t}\mathbf{I}
> \right)
> $$
>
> 두 번째 항:  
>
> $$
> q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)
> = \mathcal{N}\!\left(
> \mathbf{x}_{t-1};
> \mu_2 = \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0,
> \Sigma_2 = (1 - \bar{\alpha}_{t-1})\mathbf{I}
> \right)
> $$
>
> 이제 가우시안 곱 정리를 적용하면  
> 사후분포 $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ 는  
> 또 다른 가우시안 형태로 표현된다:
>
> $$
> q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)
> = \mathcal{N}\!\left(
> \mathbf{x}_{t-1};
> \tilde{\mu}_t, \tilde{\Sigma}_t
> \right)
> $$
>
> ---
>
> **(1) 공분산 행렬의 계산**
>
> 가우시안 곱 정리에 따라  
> 공분산은 다음과 같이 주어진다:
>
> $$
> \tilde{\Sigma}_t
> = \left(\Sigma_1^{-1} + \Sigma_2^{-1}\right)^{-1}
> $$
>
> 각 항을 대입하면,
>
> $$
> \Sigma_1^{-1} = \frac{\alpha_t}{\beta_t}\mathbf{I}, \quad
> \Sigma_2^{-1} = \frac{1}{1 - \bar{\alpha}_{t-1}}\mathbf{I}
> $$
>
> 따라서
>
> $$
> \tilde{\Sigma}_t
> = \left(
> \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}
> \right)^{-1}\mathbf{I}
> $$
>
> 분모를 통분하면,
>
> $$
> \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}
> = \frac{\alpha_t(1 - \bar{\alpha}_{t-1}) + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})}
> $$
>
> 여기서 $\beta_t = 1 - \alpha_t$ 이므로,
>
> $$
> \alpha_t(1 - \bar{\alpha}_{t-1}) + \beta_t
> = \alpha_t - \alpha_t\bar{\alpha}_{t-1} + 1 - \alpha_t
> = 1 - \alpha_t\bar{\alpha}_{t-1}
> = 1 - \bar{\alpha}_t
> $$
>
> 따라서,
>
> $$
> \tilde{\Sigma}_t
> = \frac{\beta_t(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\mathbf{I}
> = \tilde{\beta}_t \mathbf{I}
> $$
>
> 즉, 분산 항은 다음과 같이 정리된다.
>
> $$
> \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\beta_t
> $$
>
> ---
>
> **(2) 평균 벡터의 계산**
>
> 평균은 다음 식으로 계산된다:
>
> $$
> \tilde{\mu}_t
> = \tilde{\Sigma}_t
> \left(
> \Sigma_1^{-1}\mu_1 + \Sigma_2^{-1}\mu_2
> \right)
> $$
>
> 각 항을 대입하면,
>
> $$
> \Sigma_1^{-1}\mu_1 + \Sigma_2^{-1}\mu_2
> = \frac{\alpha_t}{\beta_t}\frac{\mathbf{x}_t}{\sqrt{\alpha_t}}
> + \frac{1}{1 - \bar{\alpha}_{t-1}}\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0
> $$
>
> $$
> = \frac{\sqrt{\alpha_t}}{\beta_t}\mathbf{x}_t
> + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}}\mathbf{x}_0
> $$
>
> 이를 다시 $\tilde{\Sigma}_t$ 와 곱하면:
>
> $$
> \tilde{\mu}_t
> = \frac{\beta_t(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}
> \left(
> \frac{\sqrt{\alpha_t}}{\beta_t}\mathbf{x}_t
> + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}}\mathbf{x}_0
> \right)
> $$
>
> 분배법칙을 적용하면,
>
> $$
> \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0)
> =
> \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}\mathbf{x}_0
> + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\mathbf{x}_t
> $$
>
> ---
>
> **(3) 결과**
>
> 최종적으로 사후분포는 다음과 같이 정리된다:
>
> $$
> q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)
> =
> \mathcal{N}\!\left(
> \mathbf{x}_{t-1};
> \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0),
> \tilde{\beta}_t\mathbf{I}
> \right)
> $$
>
> 여기서
> $$\tilde{\beta}_t$$와 $$\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0)$$는 위에서 정의한 형태이다.

따라서 식 (5)의 모든 KL 발산은  
가우시안 분포 간의 비교로 표현되므로,  
고분산의 몬테카를로 추정(Monte Carlo estimates) 대신  
폐형식(closed-form) 표현을 이용한  
Rao-Blackwellization 방식으로 계산할 수 있다.

> 위 문장의 의미를 구체적으로 설명하면 다음과 같다.  
>
> 식 (5)에서 등장하는 모든 KL 발산 항  
>
> $$
> D_{\mathrm{KL}}\big(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)
> \parallel
> p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)\big)
> $$
>
> 은 두 확률분포가 모두 가우시안 형태임을 이용해  
> 닫힌 해(closed-form)로 계산할 수 있다.  
>
> 즉, 두 분포가 모두  
>
> $$
> q = \mathcal{N}(\mu_q, \Sigma_q), \quad
> p = \mathcal{N}(\mu_p, \Sigma_p)
> $$
>
> 와 같은 정규분포일 때,  
> KL 발산은 다음의 분석적 식으로 바로 계산된다.
>
> $$
> D_{\mathrm{KL}}(q \parallel p)
> = \frac{1}{2}
> \left[
> \mathrm{tr}(\Sigma_p^{-1}\Sigma_q)
> + (\mu_p - \mu_q)^\top \Sigma_p^{-1}(\mu_p - \mu_q)
> - k
> + \log\frac{\det\Sigma_p}{\det\Sigma_q}
> \right]
> $$
>
> 이처럼 KL 발산을 수치적 샘플링(몬테카를로 추정)을 통해 근사하지 않고  
> 수학적으로 정확한 폐형식(closed-form expression)으로 계산할 수 있기 때문에,  
> 추정 과정에서 발생하는 확률적 노이즈(variance)가 크게 줄어든다.  
>
> 이를 Rao–Blackwellization이라 부르는데,  
> 이는 “기댓값의 분산을 줄이기 위해 조건부 기댓값 형태로 재작성하는 방법”을 의미한다.  
> 즉, $E[f(X)]$ 를 단순 샘플링으로 추정하는 대신  
> $E[E[f(X)\mid Y]]$ 와 같이 조건부 기댓값을 이용해  
> 더 안정적이고 분산이 낮은 추정치를 얻는 것이다.  
>
> 요약하면,  
> 식 (5)의 각 KL 항이 가우시안 간의 닫힌 해로 표현됨으로써  
> 학습 과정에서 몬테카를로 샘플링에 의존하지 않아도 되며,  
> 그 결과 ELBO 기반 학습보다 분산이 낮고 안정적인 최적화가 가능해진다.
