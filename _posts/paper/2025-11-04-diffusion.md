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

