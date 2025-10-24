---
layout: post
title: "[1. Introduction] An Introduction to Variational Autoencoders"
date: 2025-10-22 20:00:00 +0900
categories:
  - "Books"
  - "An Introduction to Variational Autoencoders"
tags: []
---

# 1. Introduction

## 1.1 동기 (Motivation)

머신러닝은 주로 생성(generative)모델링과 판별(discriminative) 모델링로 구분된다.  

판별 모델링에서는 관측된 데이터가 주어졌을 때 예측기(predictor)를 학습시키는 것이 목표인 반면,  
생성 모델링에서는 모든 변수들에 대한 결합 분포를 학습하는 것과 같이 보다 일반적인 문제를 다루는 것이 목표이다.  

생성 모델은 현실 세계에서 데이터가 어떻게 생성되는지를 모사(simulate)한다.  

‘모델링(modeling)’이라는 개념은 거의 모든 과학 분야에서  
"생성 과정을 밝히기 위해 이론을 가정하고 관측을 통해 이를 검증하는 과정"으로 이해된다.  

예를 들어, 기상학자가 날씨를 모델링할 때는 날씨의 근본적인 물리 법칙을 표현하기 위해 매우 복잡한 편미분방정식을 사용한다.  
또한 천문학자가 은하의 형성을 모델링할 때는 별들 간의 상호작용을 지배하는 물리 법칙을 운동 방정식에 담는다.  

이와 같은 원리는 생물학자, 화학자, 경제학자 등 다른 과학자들에게도 동일하게 적용된다.  

즉, 과학에서의 모델링은 사실상 거의 언제나 생성 모델링인 것이다.

---

생성 모델링이 매력적인 이유는 많다.  

첫째, 우리는 물리 법칙과 제약 조건을 생성 과정에 포함시킬 수 있으며,  
우리가 알지 못하거나 관심 없는 세부 사항들, 즉 성가신 변수들은 노이즈로 취급된다.  

이러한 방식으로 얻어진 모델들은 일반적으로 매우 직관적이고 해석 가능하며,  
이를 관측값과 대조함으로써 우리는 세상이 어떻게 작동하는지에 대한 우리의 이론을 확인하거나 기각할 수 있다.

---

데이터의 생성 과정을 이해하려고 하는 또 다른 이유는  
그 과정이 자연스럽게 세상의 인과(causal) 관계를 표현하기 때문이다.  

인과 관계는 단순한 상관관계(correlations)보다 새로운 상황에 훨씬 더 잘 일반화된다는 큰 이점을 가진다.  

예를 들어, 우리가 지진의 생성 과정을 이해하게 되면, 그 지식을 캘리포니아나 칠레에서 모두 활용할 수 있다.

---

'생성 모델'을 '판별기'로 전환하려면 베이즈 규칙(Bayes rule)을 사용해야 한다.  

예를 들어, 우리가 A 유형의 지진과 B 유형의 지진에 대한 각각의 생성 모델을 가지고 있다면,  
이 두 모델 중 어느 쪽이 주어진 데이터를 더 잘 설명하는지를 비교함으로써,  
지진 A가 발생했을 확률과 지진 B가 발생했을 확률을 계산할 수 있다.  

그러나 베이즈 규칙을 적용하는 과정은 계산적으로 매우 비용이 많이 드는 경우가 많다.

---

판별적 방법에서는 미래 예측을 수행하려는 방향과 동일한 방향으로의 사상(map)을 직접 학습한다.  
이는 생성 모델의 방향과는 반대이다.  

예를 들어, 세상에서 이미지가 생성되는 과정을 생각해보면,  
먼저 객체를 인식(identify)하고, 그 다음 객체를 3차원(3D)으로 생성(generate)한 뒤,  
이를 픽셀 격자(pixel grid)에 투영(project)한다고 볼 수 있다.  

반면, 판별 모델은 이러한 픽셀 값들을 직접 입력으로 받아, 레이블(labels)로 매핑한다.  

생성 모델은 데이터로부터 효율적으로 학습할 수 있지만,  
순수한 판별 모델에 비해 데이터에 대해 더 강한 가정을 세우는 경향이 있다.  
그 결과, 모델이 잘못되었을 경우 점근적 편향(asymptotic bias)이 더 크게 나타나는 경우가 있다 (Banerjee, 2007).  

> 여기서 ‘점근적 편향(asymptotic bias)’은 데이터의 양이 무한히 많아져도 사라지지 않는  
> 체계적 오차(systematic error)를 의미한다.  
> 즉, 모델이 구조적으로 잘못 설정되어 있다면,  
> 아무리 많은 데이터를 학습하더라도 진짜 분포에 완전히 수렴하지 못한다는 뜻이다. 

따라서 모델이 잘못된 경우 — 그리고 실제로 대부분 어느 정도는 잘못되어 있다! —  
단순히 판별만을 학습하는 것이 목적이며, 충분히 많은 데이터가 주어진 환경에 있다면,  
순수한 판별 모델이 판별 과제에서 더 적은 오류를 낳는 경향을 보인다.  

그럼에도 불구하고, 데이터의 양에 따라,  
생성 과정을 연구하는 것이 판별기 — 예를 들어 분류기(classifier) — 의 학습에 도움이 될 수 있다.  

예를 들어, 레이블이 있는 데이터는 적고, 레이블이 없는 데이터는 훨씬 많은 경우,  
즉 준지도 학습(semi-supervised learning) 환경에서는,  
데이터의 생성 모델을 이용하여 분류성능을 향상시킬 수 있다 (Kingma et al., 2014; Sønderby et al., 2016a).

---

생성 모델링은 보다 일반적으로 유용할 수 있다.  

이를 보조(auxiliary) 작업으로 생각할 수도 있다.  

예를 들어, 가까운 미래를 예측하는 것은 세상을 이해하기 위한 유용한 추상적 표현을 형성하게 하며,  
이러한 표현은 이후 여러 예측 과제들에서 활용될 수 있다. 

데이터의 변화 요인(factors of variation) 가운데  
분리되어 있으며(disentangled), 의미론적으로 타당하고(semantically meaningful),  
통계적으로 독립적이며(statistically independent), 인과적인(causal) 요소들을 찾는 이 탐구는  
일반적으로 비지도 표현 학습(unsupervised representation learning)으로 알려져 있으며,  
이를 위해 변분 오토인코더(variational autoencoder, VAE)가 널리 활용되어 왔다.  

또 다른 관점에서는 이를 일종의 암묵적 정규화(implicit regularization)로 볼 수 있다.  
즉, 표현이 데이터 생성 과정에서 의미 있게 작동하도록 강제함으로써,  
입력에서 표현으로 가는 역방향 과정이 일정한 구조 안으로 유도되도록 만드는 것이다.  

>오토인코더 구조에서 생성 모델은 표현으로부터 입력을 생성하므로,  
>입력에서 표현으로 가는 인코딩 과정은 생성 과정의 역방향으로 볼 수 있다.

이러한 보조 작업, 즉 세상을 예측하는 작업은 세계를 추상적인 수준에서 더 잘 이해하기 위해 사용되며,  
결국 후속 예측(downstream prediction)을 더 잘 수행할 수 있도록 돕는다.

---

변분 오토인코더(VAE)는 서로 연결되어 있지만, 각각 독립적으로 매개변수화된 두 개의 모델로 볼 수 있다.  

즉, 인코더(encoder) 또는 인식 모델(recognition model)과  
디코더(decoder) 또는 생성 모델(generative model)이다.  

이 두 모델은 서로를 보완하며 협력한다.  

인식 모델은 잠재 확률 변수(latent random variables)에 대한  
사후분포(posterior)의 근사값을 생성 모델에 제공한다.  

> 인식 모델은 입력 변수 x를 받아 잠재 확률 변수 z를 생성한다.  
> 즉, 데이터 x를 관측한 이후의 z에 대한 확률 분포를 추정하므로,  
> 이는 z에 대한 사후분포(posterior)로 볼 수 있다.

생성 모델은 이를 이용하여 기대-최대화(expectation–maximization, EM) 학습의  
한 반복(iteration) 내에서 자신의 매개변수를 갱신한다.  

> 인식 모델이 제공한 사후분포 근사값 $q_\phi(z \mid x)$ 은  
> EM 알고리즘의 E단계(expectation step)에 해당하며,  
> 생성 모델은 이를 바탕으로 M단계(maximization step)에서  
> 데이터의 우도(likelihood)를 최대화한다.

반대로, 생성 모델은, 인식 모델이 데이터의 의미 있는 표현(예를 들어 클래스 레이블)을 학습할 수 있도록  
일종의 학습 틀(scaffolding) 역할을 한다.  

즉, 인식 모델은 베이즈 규칙에 따라 생성 모델의 근사적 역함수로 작동한다.

---

일반적인 변분 추론(Variational Inference, VI)과 비교했을 때,  
변분 오토인코더(VAE) 프레임워크의 한 가지 장점은  
인식 모델(또는 추론 모델)이 입력 변수의 (확률적) 함수로 정의된다는 점이다.  

> 기존 VI에서는 데이터 샘플마다 각각 다른 분포를 따로 학습해야 하지만,  
> VAE에서는 하나의 인식 모델이 모든 데이터에 대해 공통으로 작동하므로  
> 훨씬 효율적으로 추론을 수행할 수 있다.

이는 VI에서 각 데이터 샘플마다 별도의 변분 분포(variational distribution)를 두는 방식과 대조된다.  
그러한 접근은 데이터셋이 커질수록 비효율적이다.  

VAE의 인식 모델은 하나의 파라미터 집합으로 입력 변수와 잠재 변수 간의 관계를 학습하며,  
이러한 방식을 상각 추론(amortized inference)이라고 부른다.  

이 인식 모델은 임의의 복잡한 형태를 가질 수 있지만,  
구조상 입력에서 잠재 변수로의 단일 feedforward 연산만으로 수행되므로 비교적 빠르게 계산할 수 있다.  

그러나 그 대가로, 이러한 샘플링 과정은  
학습에 필요한 그래디언트에 샘플링 노이즈(sampling noise)를 유발한다.  

VAE 프레임워크의 가장 큰 공헌 중 하나는 이러한 분산(variance)을 줄이기 위해  
재매개변수화 기법(reparameterization trick)으로 알려진 간단한 그래디언트 계산 재구성 절차를 도입했다는 점이다.  
이를 통해 그래디언트의 분산을 효과적으로 줄일 수 있다.

---

변분 오토인코더(VAE)는 헬름홀츠 머신(Helmholtz Machine, Dayan et al., 1995)에서 영감을 받았다.  
헬름홀츠 머신은 아마도 최초로 인식 모델(recognition model)을 도입한 모델이었다.  

그러나 그 학습 방식인 wake-sleep 알고리즘은 비효율적이었으며, 단일 목적 함수를 최적화하는 방식이 아니었다.  

> 헬름홀츠 머신은 생성 모델과 인식 모델을 번갈아 학습했지만,  
> VAE는 하나의 명확한 목적 함수(ELBO)를 중심으로  
> 두 모델을 함께 최적화한다는 점에서 다르다.

반면, VAE의 학습 규칙은  
최대우도(maximum likelihood) 목표를 단일 근사 형태로 유도하여  
보다 일관되고 효율적인 학습이 가능하도록 한다.  

---

변분 오토인코더(VAE)는 그래픽(graphical) 모델과 딥러닝을 결합한 구조이다.  

생성 모델은 $p(x \mid z)p(z)$과 같은 형태의 베이지안 네트워크(Bayesian network)로 표현된다:  
  
또는 다층 잠재 변수가 존재하는 경우,  
$p(x \mid z_L)p(z_L \mid z_{L-1}) \dots p(z_1 \mid z_0)$ 와 같은 계층적(hierarchical) 구조를 가진다.  

마찬가지로 인식 모델(추론 모델)도 $q(z \mid x)$ 형태의 조건부 베이지안 네트워크이거나,  
$q(z_0 \mid z_1) \dots q(z_L \mid x)$ 와 같은 계층 구조로 표현될 수 있다.  

이때 각 조건부 분포 안에는 $z \mid x \sim f(x, \epsilon)$ 과 같이 복잡한 딥러닝 신경망이 내포되어 있을 수 있다.   
여기에서 $f$ 는 신경망 기반의 매핑 함수이며, $\epsilon$ 은 노이즈이다.  

VAE의 학습 알고리즘은  
고전적인 변분 기대-최대화(variational expectation–maximization)를 기반으로 하지만,  
재매개변수화 기법(reparameterization trick)을 통해  
그 안에 포함된 여러 신경망 계층을 역전파(backpropagation)로 학습한다.  

---

VAE 프레임워크는 제안된 이후 다양한 방향으로 확장되어 왔다.  
예를 들어, 동적 모델(dynamical models) (Johnson et al., 2016),  
어텐션(attention)이 포함된 모델 (Gregor et al., 2015),  
다층 확률 잠재 변수(multiple levels of stochastic latent variables)를 가지는 모델 (Kingma et al., 2016) 등이 있다.  

이처럼 VAE는 새로운 생성 모델을 설계하기 위한 풍부한 기반(fertile framework)으로 자리잡았다.  

최근에는 또 다른 생성 모델링 패러다임인  
적대적 생성 신경망(Generative Adversarial Network, GAN)(Goodfellow et al., 2014) 이  
큰 주목을 받게 되었다.  

VAE와 GAN은 상호 보완적인 특성을 가진다.  
GAN은 주관적 지각 품질(subjective perceptual quality)이 높은 이미지를 생성할 수 있지만,  
데이터 전체 분포를 충분히 포괄하지 못한다 (Grover et al., 2018).  
반면, 우도 기반(likelihood-based) 생성 모델인 VAE는  
샘플이 상대적으로 더 퍼져 보이지만,  
우도 기준의 관점에서는 더 나은 밀도 모델이다.  

> GAN은 시각적으로 사실적인 이미지를 잘 생성하지만,  
> 데이터 분포 전체를 충분히 학습하지 못해 일부 영역을 놓치는 경향이 있다.  
> 반대로 VAE는 데이터의 전체 확률 구조를 잘 포착하지만,  
> 생성된 샘플이 다소 흐릿하게 보이는 trade-off가 존재한다.

이러한 이유로,  
두 접근법의 장점을 결합하려는 다양한 하이브리드 모델(hybrid models)들이 제안되어 왔다  
(Dumoulin et al., 2017; Grover et al., 2018; Rosca et al., 2018).  

---

우리 연구 공동체는 이제 생성 모델과 비지도 학습이 지능형 시스템을 구축하는 데  
중요한 역할을 한다는 사실을 받아들이고 있는 듯하다.  

우리는 VAE가 그 퍼즐의 한 조각으로서 유용한 기여를 하길 기대한다.  

---

## 1.2 목표 (Aim)

변분 오토인코더(VAE, Variational Autoencoder) 프레임워크 (Kingma & Welling, 2014; Rezende et al., 2014)는  
심층 잠재 변수 모델(deep latent-variable model)과 그에 대응하는 추론 모델(inference model)을  
확률적 경사 하강법(stochastic gradient descent)을 통해 동시에 학습할 수 있는 체계적인 방법을 제공한다.  

이 프레임워크는 생성 모델링(generative modeling), 준지도 학습(semi-supervised learning),  
표현 학습(representation learning) 등 다양한 영역에 걸쳐 폭넓게 활용된다.

---

이 글은 이전 연구 (Kingma & Welling, 2014)의 확장판으로,  
주제를 더 세밀하게 설명하고 이후의 주요 후속 연구들을 함께 논의하기 위한 것이다.  

모든 관련 연구를 포괄하는 종합적인 리뷰를 목표로 하지는 않는다.  
독자가 대수학(algebra), 미적분학(calculus),  
그리고 확률론(probability theory)에 대한 기본적인 지식을 갖추고 있다고 가정한다.

---

이 장에서는 다음과 같은 배경 내용을 다룬다.  

확률 모델(probabilistic models), 유향 그래픽 모델(directed graphical models),  
그리고 유향 그래픽 모델과 신경망(neural networks)의 결합,  

또한 완전 관측 모델(fully observed models)과  
심층 잠재 변수 모델(Deep Latent-Variable Models, DLVMs)에서의 학습 방법을 살펴본다.  

2장에서는 VAE의 기본 개념을,  
3장에서는 고급 추론 기법(advanced inference techniques)을,  
4장에서는 고급 생성 모델(advanced generative models)을 설명한다.  

수학적 표기법에 대한 보다 자세한 내용은  
부록 A.1 절(section A.1)을 참고하라.

---

## 1.3 확률 모델과 변분 추론(Variational Inference)

머신러닝 분야에서는 다양한 자연적 또는 인공적 현상에 대해  
데이터로부터 확률 모델(probabilistic model)을 학습하는 데 관심이 많다.  

확률 모델은 이러한 현상을 수학적으로 기술한 표현이다.  
이들은 현상을 이해하거나, 미래의 미지값(unknowns)을 예측하거나,  
보조적 또는 자동화된 의사결정을 수행하는 데 유용하다.  

즉, 확률 모델은 지식(knowledge)과 기술(skill)의 개념을 형식화(formalize)하며,  
머신러닝과 인공지능(AI) 분야의 핵심적인 구성 요소로 자리한다.

---

확률 모델에는 일반적으로 미지의 요소(unknowns)가 포함되어 있으며,  
데이터만으로는 이러한 미지의 요소를 완전히 설명하기 어렵다.  
따라서 모델의 일부 측면에 대해 일정 수준의 불확실성(uncertainty)을 가정해야 한다.  

이러한 불확실성의 정도와 속성은 (조건부) 확률분포의 관점에서 정의된다.  

모델은 연속형 변수(continuous-valued variables)와  
이산형 변수(discrete-valued variables)를 모두 포함할 수 있다.  

가장 완전한 형태의 확률 모델은  
모델 내의 변수들 사이의 모든 상관관계와 고차 의존성(higher-order dependencies)을,  
그 변수들에 대한 결합 확률분포(joint probability distribution)로 명시한다.

---

벡터 $ \mathbf{x} $ 를, 우리가 모델링하고자 하는 모든 관측 변수들의 집합을 나타내는 벡터로 사용하자.  
표기법을 단순화하고 혼잡함을 피하기 위해,  
소문자 굵은체 (예: $ \mathbf{x} $) 를 관측된 확률변수들의 집합을 나타내는 기호로 사용한다.  
이는 해당 변수들이 평탄화(flattened)되고 연결(concatenated)되어  
하나의 벡터로 표현된다는 의미이다.  

자세한 표기법에 대해서는 A.1 절을 참고하라.

---

우리는 관측 변수 $ \mathbf{x} $ 가  
알려지지 않은 생성 과정(unknown underlying process)으로부터 추출된  
무작위 표본(random sample)이라고 가정한다.  

이 과정의 실제 (확률) 분포 $ p^{*}(\mathbf{x}) $ 는 알려져 있지 않다.  

따라서 우리는 이 근본적인 과정을  
파라미터 $ \boldsymbol{\theta} $ 를 갖는 선택된 모델 $ p_{\boldsymbol{\theta}}(\mathbf{x}) $ 로 근사하려고 한다.

$$
\mathbf{x} \sim p_{\boldsymbol{\theta}}(\mathbf{x})
\tag{1.1}
$$

---

학습(Learning)이란, 일반적으로  
모델이 정의하는 확률분포 $ p_{\boldsymbol{\theta}}(\mathbf{x}) $ 가 데이터의 실제 분포 $ p^{*}(\mathbf{x}) $ 를 근사하도록 하는  
모수(parameter) $ \boldsymbol{\theta} $ 의 값을 찾는 과정이다.  

즉, 임의의 관측값 $ \mathbf{x} $ 에 대해 다음이 성립하도록 하는 것이다:

$$
p_{\boldsymbol{\theta}}(\mathbf{x}) \approx p^{*}(\mathbf{x})
\tag{1.2}
$$

---

당연하게도, 우리는 모델 $ p_{\boldsymbol{\theta}}(\mathbf{x}) $ 가  
데이터에 적응할 수 있을 만큼 충분히 유연(flexible)하여  
충분히 정확한 모델을 얻을 가능성을 가지길 원한다.  

동시에, 사전에 알고 있는(priori) 데이터 분포에 대한 지식(knowledge)을  
모델에 반영할 수 있기를 바란다.

---

### 1.3.1 조건부(Conditional) 모델

분류(classification)나 회귀(regression) 문제의 경우처럼,  
우리는 비(非)조건부 모델 $ p_{\boldsymbol{\theta}}(\mathbf{x}) $ 을 학습하는 데 관심이 있는 것이 아니라,  
조건부 모델 $ p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x}) $ 을 학습하는 데 관심이 있다.  

이 모델은 실제 데이터가 따르는 조건부 분포 $ p^{*}(\mathbf{y} \mid \mathbf{x}) $ 를 근사(approximate)한다.  
즉, 관측된 변수 $ \mathbf{x} $ 의 값에 따라 변수 $ \mathbf{y} $ 의 확률 분포를 정의하는 것이다.  

이때 $ \mathbf{x} $ 는 흔히 모델의 입력(input)이라고 불린다.  

비(非)조건부 모델의 경우와 마찬가지로,  
모델 $ p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x}) $을 선택하고 알려지지 않은 실제 분포에 근접하도록 최적화한다.  
즉, 모든 $ \mathbf{x} $ 와 $ \mathbf{y} $ 에 대해 다음이 성립하도록 한다.

$$
p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x}) \approx p^{*}(\mathbf{y} \mid \mathbf{x})
\tag{1.3}
$$

---

조건부 모델링의 대표적이고 간단한 예로는 이미지 분류(image classification)가 있다.  
이 경우 $ \mathbf{x} $ 는 이미지이고, $ \mathbf{y} $ 는 사람이 부여한 '이미지의 클래스',  
즉 우리가 예측하고자 하는 라벨(label)이다.  

이때 모델 $ p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x}) $는 일반적으로 범주형 분포(categorical distribution)로 설정되며,  
그 분포의 파라미터는 입력 이미지 $ \mathbf{x} $ 로부터 계산된다.

---

예측해야 하는 변수가 매우 고차원(high-dimensional)일 때,  
예를 들어 이미지, 영상, 음성과 같은 경우에는 조건부 모델을 학습하기가 훨씬 더 어렵다.  

한 가지 예는 이미지 분류 문제의 반대 경우이다.  
즉, 클래스 레이블이 주어졌을 때, 그 조건에 따라 이미지의 확률 분포를 예측하는 문제이다.

또 입력과 출력이 모두 고차원인 예로는, 텍스트나 영상 예측과 같은 시계열 예측 문제가 있다.

---

표기상의 복잡함을 피하기 위해, 이 글에서는 종종 비(非)조건부 모델링(unconditional modeling)을 가정한다.  

그러나 여기서 소개하는 방법들은 거의 모든 경우에  
조건부 모델(conditional models)에도 동일하게 적용될 수 있다는 점을 기억해야 한다.  

조건부 모델에서의 데이터는 모델의 입력(input)으로 간주할 수 있으며,  
이는 모델의 매개변수(parameter)와 유사하지만  
그 값(value)에 대해 최적화를 수행하지 않는다는 점에서 차이가 있다.  

---

## 1.4 신경망을 이용한 조건부 분포의 파라미터화 (Parameterizing Conditional Distributions with Neural Networks)

미분 가능한 '피드포워드(feed-forward) 신경망' (이후 간단히 신경망이라 부르기로 한다)은  
매우 유연하고 계산적으로 확장 가능한 형태의 함수 근사기(function approximator)이다.  

여러 개의 ‘은닉층(hidden layers)’을 갖는 신경망(neural networks)을 학습하는 것은  
일반적으로 딥러닝(deep learning) (Goodfellow et al., 2016; LeCun et al., 2015)이라고 불린다.  

특히 흥미로운 응용 분야 중 하나는 확률 모델(probabilistic models)이다.  
즉, 신경망을 확률밀도함수(probability density functions, PDFs)  
또는 확률질량함수(probability mass functions, PMFs)의 형태로 활용하는 것이다.  

신경망 기반의 확률 모델은  
확률적 그래디언트 기반 최적화(stochastic gradient-based optimization)를 사용할 수 있기 때문에  
계산적으로 확장 가능하다.  

이러한 특성은 이후에 설명하겠지만, 대규모 모델과 대규모 데이터셋으로의 확장을 가능하게 한다.

깊은 신경망(deep neural network)은 벡터 함수 형태로 다음과 같이 표기한다: $$\text{NeuralNet}(\cdot)$$

---

작성 시점을 기준으로, 딥러닝은 다양한 분류 및 회귀 문제에서 우수한 성능을 보이는 것으로 확인되었다.  
(LeCun et al., 2015; Goodfellow et al., 2016).  

예를 들어, 신경망 기반 이미지 분류(LeCun et al., 1998)의 경우,  
신경망은 클래스 레이블 $ y $ 에 대한 범주형 분포 $ p_{\boldsymbol{\theta}}(y \mid \mathbf{x}) $ 를 이미지 $ \mathbf{x} $ 에 조건화하여 파라미터화한다.

$$
\mathbf{p} = \text{NeuralNet}(\mathbf{x})
\tag{1.4}
$$

$$
p_{\boldsymbol{\theta}}(y \mid \mathbf{x}) = \text{Categorical}(y; \mathbf{p})
\tag{1.5}
$$

여기서 NeuralNet(·)의 마지막 연산은 일반적으로 소프트맥스(softmax) 함수이며,  
이를 통해 다음 조건이 만족된다: $$\sum_i p_i = 1$$

---

## 1.5 유향 그래픽 모델(Directed Graphical Models)과 신경망

우리는 유향 확률 모델(directed probabilistic models)을 다룬다.  
이들은 유향 확률 그래픽 모델(directed probabilistic graphical models, PGMs)  
또는 베이지안 네트워크라고도 불린다.  

유향 그래픽 모델은 모든 변수가 유향 비순환 그래프(directed acyclic graph, DAG)로  
위상적으로(topologically) 구성된 형태의 확률 모델이다.  

이러한 모델의 변수들에 대한 결합 분포(joint distribution)는  
사전분포(prior)와 조건부분포(conditional distribution)의 곱 형태로 분해된다.

$$
p_{\boldsymbol{\theta}}(\mathbf{x}_1, \dots, \mathbf{x}_M)
= \prod_{j=1}^{M} p_{\boldsymbol{\theta}}(\mathbf{x}_j \mid Pa(\mathbf{x}_j))
\tag{1.6}
$$

여기서 $ Pa(\mathbf{x}_j) $ 는 그래프에서 노드 $ j $ 의 부모 변수(parent variables) 집합을 의미한다.  
루트 노드(root node)가 아닌 경우, 부모 노드에 조건화된 분포를 사용한다.  
루트 노드의 경우에는 부모 집합이 공집합이므로, 그 분포는 비(非)조건부(unconditional) 분포가 된다.

---

전통적으로, 각 조건부 확률분포 $ p_{\boldsymbol{\theta}}(\mathbf{x}_j \mid Pa(\mathbf{x}_j)) $ 는  
룩업 테이블(lookup table) 또는 선형 모델(linear model)로 파라미터화되었다 (Koller and Friedman, 2009).  

> 과거에는 조건부 확률분포를 표현하기 위해  
> 각 변수의 모든 가능한 입력 조합을 나열한 룩업 테이블이나, 단순한 선형 관계를 가정하는 모델이 주로 사용되었다.  
> 그러나 이러한 방식은 고차원 데이터나 복잡한 비선형 관계를 표현하기 어렵다는 한계가 있다.

앞서 설명했듯이, 이러한 조건부 분포를 더 유연하게 파라미터화하는 방법은 신경망을 사용하는 것이다.  

이 경우, 신경망은 유향 그래프에서 변수의 부모 노드를 입력으로 받아 그 변수에 대한 '분포의 파라미터' $ \boldsymbol{\eta} $ 를 출력한다.

$$
\boldsymbol{\eta} = \text{NeuralNet}(Pa(\mathbf{x}))
\tag{1.7}
$$

$$
p_{\boldsymbol{\theta}}(\mathbf{x} \mid Pa(\mathbf{x})) = p_{\boldsymbol{\theta}}(\mathbf{x} \mid \boldsymbol{\eta})
\tag{1.8}
$$

이제 모든 변수가 데이터에서 관측된 경우, 이러한 모델들의 파라미터를 어떻게 학습할 것인지 살펴볼 것이다.

---

## 1.6 완전 관측 모델(Fully Observed Models)에서의 신경망 학습

유향 그래픽 모델의 모든 변수가 데이터에서 관측된다면,  
그 모델 하에서의 데이터 로그 확률(log-probability)을 계산하고 이에 대해 미분할 수 있다.  
따라서 비교적 직접적이고 명확한 방식으로 최적화(optimization)를 수행할 수 있다.  

---

### 1.6.1 데이터셋 (Dataset)

보통 우리는 $ N \ge 1 $ 개의 데이터 포인트로 구성된 데이터셋 $ \mathcal{D} $ 를 수집한다.

$$
\mathcal{D} = \{ \mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(N)} \}
\equiv \{ \mathbf{x}^{(i)} \}_{i=1}^{N}
\equiv \mathbf{x}^{(1:N)}
\tag{1.9}
$$

각 데이터 포인트들은 변하지 않는(unchanging) 기저 분포(underlying distribution)로부터  
독립적으로 추출된 표본(independent samples)이라고 가정한다.  

다시 말해, 데이터셋은 동일한(변하지 않는) 시스템으로부터 서로 독립적인 관측값들로 구성된다고 본다.  

이 경우, 관측값 집합 $ \mathcal{D} = \{ \mathbf{x}^{(i)} \}_{i=1}^{N} $는 i.i.d. (independently and identically distributed),  
즉 독립적이며 동일 분포를 따르는 표본으로 간주된다.  

i.i.d. 가정하에서, 파라미터가 주어졌을 때의 데이터 포인트들의 확률은  
각 데이터 포인트의 확률들의 곱으로 표현된다.  
따라서 모델이 데이터에 할당하는 로그 확률은 다음과 같다.

$$
\log p_{\boldsymbol{\theta}}(\mathcal{D})
= \sum_{\mathbf{x} \in \mathcal{D}} \log p_{\boldsymbol{\theta}}(\mathbf{x})
\tag{1.10}
$$

---

### 1.6.2 최대우도와 미니배치 확률적 경사하강법 (Maximum Likelihood and Minibatch SGD)

확률 모델에서 가장 일반적으로 사용되는 기준은 최대 로그우도(maximum log-likelihood, ML)이다.  

이후에 설명하겠지만, 로그우도 기준을 최대화하는 것은  
데이터 분포와 모델 분포 간의 쿨백–라이블러 발산(Kullback–Leibler divergence, KL divergence)을  
최소화하는 것과 동등하다.

---

최대우도(ML) 기준에서는,  
모델이 데이터에 할당한 로그 확률(log-probability)의 합(또는 평균)을 최대화하는 파라미터 $ \boldsymbol{\theta} $ 를 찾는 것이 목표이다.  

i.i.d. 데이터셋 $ \mathcal{D} $ 의 크기가 $ N_D $ 일 때,  
최대우도 목적함수(maximum likelihood objective)는 식 (1.10)에서 정의된 로그 확률을 최대화하는 것으로 표현된다.

---

미적분학의 연쇄 법칙(chain rule)과 자동미분(automatic differentiation) 도구를 사용하면,  
이 목적함수(objective)의 그래디언트(gradient),  
즉 파라미터 $ \boldsymbol{\theta} $ 에 대한 1차 미분을 효율적으로 계산할 수 있다.  

이러한 그래디언트를 이용해 반복적으로 언덕 오르기(hill-climbing)를 수행하면  
최대우도(ML) 목적함수의 지역 최적점(local optimum)에 도달할 수 있다.  

만약 모든 데이터 포인트를 사용하여 $ \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(\mathcal{D}) $ 를 계산한다면,  
이를 배치 경사하강법(batch gradient descent)이라고 한다.  

그러나 이러한 미분 계산은  
데이터셋의 크기 $ N_D $ 에 비례하여 계산 비용이 선형적으로 증가하기 때문에,  
대규모 데이터셋에서는 매우 비용이 큰 연산(expensive operation)이 된다.

---

보다 효율적인 최적화 방법은 확률적 경사하강법(stochastic gradient descent, SGD)이다 (section A.3).  

SGD는 전체 데이터셋 $ \mathcal{D} $ 중 일부를 무작위로 선택한 미니배치 $ \mathcal{M} \subset \mathcal{D} $ 를 사용하며,  
그 크기는 $ N_{\mathcal{M}} $ 로 표시된다.  

이러한 미니배치를 이용하면 최대우도(ML) 기준의 불편 추정량(unbiased estimator)을 구성할 수 있다.

$$
\frac{1}{N_{\mathcal{D}}} \log p_{\boldsymbol{\theta}}(\mathcal{D})
\simeq
\frac{1}{N_{\mathcal{M}}} \log p_{\boldsymbol{\theta}}(\mathcal{M})
=
\frac{1}{N_{\mathcal{M}}}
\sum_{\mathbf{x} \in \mathcal{M}} \log p_{\boldsymbol{\theta}}(\mathbf{x})
\tag{1.11}
$$

---

이러한 그래디언트는 확률적 경사 기반 최적화 알고리즘(stochastic gradient-based optimizers)에 직접 사용할 수 있다.  
자세한 내용은 section A.3을 참조하라.  

요약하면, 목적함수(objective function)는  
확률적 그래디언트의 방향으로 작은 스텝(step)을 반복적으로 이동함으로써 최적화할 수 있다.

---

### 1.6.3 베이지안 추론 (Bayesian Inference)

베이지안 관점(Bayesian perspective)에서,  
최대우도(ML) 방법은 사후확률 최대화 추정(maximum a posteriori, MAP)을 통해 개선될 수 있다  
(section A.2.1 참조).  

또한 한 단계 더 나아가,  
모델 파라미터에 대한 근사 사후분포(approximate posterior distribution) 전체를  
추론(inference)하는 접근도 가능하다 (section A.1.4 참조).

---

## 1.7 심층 잠재 변수 모델(Deep Latent Variable Models)에서의 학습과 추론 

### 1.7.1 잠재 변수 (Latent Variables)

앞 절에서 다룬 완전 관측 유향 모델(fully-observed directed models)을  
잠재 변수를 포함하는 유향 모델(directed models)로 확장할 수 있다.  

> 예를 들어, 학생의 시험 점수 예측 문제를 생각해 보자.  
>  
> 완전 관측 모델(fully-observed model)은  
> 공부 시간이나 수면 시간 같은 관측 가능한 변수만을 이용해  
> 시험 점수를 예측한다.  
>  
> 반면, 잠재 변수 모델(latent variable model)은  
> 데이터에는 포함되지 않았지만  
> 실제 점수에 영향을 주는 집중력, 긴장도, 이해력 같은  
> 추상적 요인(latent factors)을 잠재 변수 $z$ 로 두고 함께 학습한다.  
>  
> 즉, 완전 관측 모델이  
> “관측된 변수들 간의 통계적 관계”만 학습한다면,  
> 잠재 변수 모델은 보이지 않는 원인(hidden causes)까지 내재화하여  
> 더 깊은 수준의 데이터 생성 구조와 일반화 능력을 학습할 수 있다.

잠재 변수란 모델의 일부이지만 관측되지 않으며, 따라서 데이터셋에는 포함되지 않는 변수를 말한다.  
보통 이러한 잠재 변수는 **$ \mathbf{z} $** 로 표기한다.  

관측 변수 **$ \mathbf{x} $** 에 대한 비(非)조건부(unconditional) 모델링의 경우,  
유향 그래픽 모델은 관측 변수 **$ \mathbf{x} $** 와 잠재 변수 **$ \mathbf{z} $** 모두에 대한 결합 분포(joint distribution) $ p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) $ 를 나타낸다.  

관측 변수에 대한 주변 분포(marginal distribution) $ p_{\boldsymbol{\theta}}(\mathbf{x}) $ 는 다음과 같이 정의된다:

$$
p_{\boldsymbol{\theta}}(\mathbf{x})
= \int p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) \, d\mathbf{z}
\tag{1.13}
$$

이 식은 단일 데이터 포인트(single datapoint)에 대한  
주변 우도(marginal likelihood) 또는 모델 증거(model evidence)라고도 불리며,  
이를 $ \boldsymbol{\theta} $ 의 함수로 볼 수 있다.

---

이와 같은 $ \mathbf{x} $ 에 대한 암묵적 분포(implicit distribution)는 상당히 유연할 수 있다.  

만약 잠재 변수 $ \mathbf{z} $ 가 이산형(discrete)이고, 조건부 분포 $ p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) $ 가 가우시안 분포라면,  
$ p_{\boldsymbol{\theta}}(\mathbf{x}) $ 는 가우시안 혼합 분포(mixture-of-Gaussians distribution)가 된다.  

> 가우시안 혼합 분포(Mixture of Gaussians, MoG)는  
> 여러 개의 가우시안 분포를 가중 평균(weighted average) 형태로 결합한 분포이다.  
>  
> 각 잠재 변수 값 $z=k$ 는 하나의 가우시안 성분(component)을 나타내며,  
> 전체 분포 $p_{\boldsymbol{\theta}}(\mathbf{x})$ 는  
> 여러 성분들이 섞인 형태로 복잡한 데이터 분포를 근사할 수 있다.  
>  
> 즉, 단일 가우시안으로는 표현하기 어려운  
> 다봉(unimodal이 아닌) 또는 비선형적 데이터 분포를  
> 보다 유연하게 모델링할 수 있게 된다.

반면 $ \mathbf{z} $ 가 연속형(continuous)인 경우,  
$ p_{\boldsymbol{\theta}}(\mathbf{x}) $ 는 사실상 무한한(infinite) 혼합 분포로 볼 수 있으며,  
이는 이산 혼합 모델보다 잠재적으로 더 강력한 표현력을 가진다.  

이러한 주변 분포(marginal distributions)는  
복합 확률분포(compound probability distributions)라고도 불린다.

> 복합 확률분포(compound probability distribution)라고 부르는 이유는,  
> 한 확률변수의 분포가 다른 확률변수의 분포에 의해 결정되기 때문이다.  
>  
> 예를 들어, 잠재 변수 $ \mathbf{z} $ 가 어떤 분포를 따르고,  
> 그 값에 따라 $ \mathbf{x} $ 의 조건부 분포 $ p(\mathbf{x} \mid \mathbf{z}) $ 가 달라진다면,  
> $ \mathbf{x} $ 의 전체 분포 $ p(\mathbf{x}) $ 는  
> 여러 분포의 혼합(compound)으로 표현된다.  
>  
> 즉, 하나의 확률변수($\mathbf{z}$)가  
> 다른 확률변수($\mathbf{x}$)의 분포 형태를 “조절”하기 때문에  
> 이러한 구조를 복합 확률분포라고 부른다.

---

### 1.7.2 심층 잠재 변수 모델 (Deep Latent Variable Models)

심층 잠재 변수 모델(Deep Latent Variable Model, DLVM)이란  
그 분포(distribution)들이 신경망으로 파라미터화된 잠재 변수 모델 $ p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) $를 의미한다.  

이러한 모델은 특정 문맥(context)에 조건화될 수도 있으며, $ p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z} \mid \mathbf{y}) $ 와 같이 표현된다.  

DLVM의 중요한 장점 중 하나는,  
유향 그래픽 모델의 각 구성 요소(사전분포나 조건부분포)가 비교적 단순한 형태(예: 조건부 가우시안 분포)라고 하더라도,  
그 주변 분포 $ p_{\boldsymbol{\theta}}(\mathbf{x}) $는 매우 복잡하고 다양한 의존성을 포함할 수 있다는 점이다.  

이러한 높은 표현력(expressivity)덕분에 DLVM은 복잡한 실제 분포 $ p^{*}(\mathbf{x}) $를 근사하는 데 매우 유용하다.  

---

아마도 가장 단순하고 가장 일반적인 DLVM은 다음과 같은 구조로 분해(factorization)되는 형태이다:

$$
p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
= p_{\boldsymbol{\theta}}(\mathbf{z}) \, p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})
\tag{1.14}
$$

여기서 $p_{\boldsymbol{\theta}}(\mathbf{z})$ 및/또는 $p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$는 직접 정의해야(specify) 한다.  

> $p_{\boldsymbol{\theta}}(\mathbf{z})$ 및/또는 $p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$는 모델 설계자가 명시적으로 정의하는 분포이며,  
> $p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})$는 이들로부터 결과적으로 도출되는 결합분포(joint distribution)이다.

분포 $p_{\boldsymbol{\theta}}(\mathbf{z})$는 어떤 관측값에도 조건화되지 않기 때문에 종종 사전분포(prior distribution)라고 불린다.

> 즉, 분포 $p_{\boldsymbol{\theta}}(\mathbf{z})$는 관측 데이터와 무관하게 정의되는 분포이다.

---

## 1.7.3 다변량 베르누이 데이터(multivariate Bernoulli data)를 위한 DLVM 예시  

이 절에서는 Kingma & Welling (2014)에서 사용된 단순한 DLVM 예시를 다룬다.  

이 모델은 이진 데이터(binary data) $\mathbf{x}$ 를 대상으로 하며,  
구형 가우시안 잠재 공간(spherical Gaussian latent space)과  
분해 가능한 베르누이 관측 모델(factorized Bernoulli observation model)을 사용한다.  

$$
p(\mathbf{z}) = \mathcal{N}(\mathbf{z}; 0, \mathbf{I})
\tag{1.15}
$$

$$
\mathbf{p} = \text{DecoderNeuralNet}_{\boldsymbol{\theta}}(\mathbf{z})
\tag{1.16}
$$

$$
\log p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})
= \sum_{j=1}^{D} \log p(x_j \mid \mathbf{z})
= \sum_{j=1}^{D} \log \text{Bernoulli}(x_j; p_j)
\tag{1.17}
$$

$$
= \sum_{j=1}^{D} \big[ x_j \log p_j + (1 - x_j)\log(1 - p_j) \big]
\tag{1.18}
$$

여기서 모든 $p_j \in \mathbf{p}$ 에 대해 $0 \le p_j \le 1$ 이며,  
이는 보통 DecoderNeuralNet의 마지막 층에서 시그모이드(sigmoid) 비선형함수로 구현된다.  
$D$는 $\mathbf{x}$의 차원 수를 의미하며,  
$\text{Bernoulli}(\cdot; p)$ 는 베르누이 분포의 확률질량함수(PMF)를 나타낸다.

> 이 예시는 Variational Autoencoder(VAE)의 기본 구조를 단순화한 형태이다.  
>  
> 잠재 변수 $ \mathbf{z} $ 는 표준정규분포 $ \mathcal{N}(0, \mathbf{I}) $ 에서 샘플링된 벡터로,  
> 데이터 생성의 ‘숨겨진 원인(latent factor)’ 역할을 한다.  
>  
> 디코더 신경망(DecoderNeuralNet)은 이 $\mathbf{z}$ 를 입력받아  
> 각 데이터 차원 $x_j$가 1이 될 확률 $p_j$를 출력한다.  
>  
> 즉, $\mathbf{p} = \text{DecoderNeuralNet}_{\boldsymbol{\theta}}(\mathbf{z})$ 는  
> “$\mathbf{z}$가 주어졌을 때 각 데이터 항목이 1일 확률”을 나타내며,  
> 마지막 층의 시그모이드(sigmoid) 함수가 이 값을 [0, 1] 범위로 제한한다.  
>  
> 분해 가능한 베르누이 관측 모델(factorized Bernoulli observation model)이란,  
> 각 데이터 차원 $x_j$가 서로 독립적이라고 가정하여  
> 전체 확률을 개별 베르누이 확률의 곱으로 표현하는 구조를 의미한다.  
>  
> 따라서 이 모델은  
> “잠재 변수 $\mathbf{z}$ → 디코더 신경망 → 베르누이 확률 $\mathbf{p}$ → 샘플된 데이터 $\mathbf{x}$”  
> 의 순서로 데이터를 생성하는 확률적 생성 과정(stochastic generation process)을 나타낸다.

## 1.8 계산 불가능성 (Intractabilities)

DLVM에서 최대우도(maximum likelihood) 학습의 주요 어려움은  
데이터의 주변확률(marginal probability) $p_{\boldsymbol{\theta}}(\mathbf{x})$ 을  
해당 모델 하에서 계산하기 어렵다는 데 있다.  

이는 식 (1.13)의 적분 항, 즉  

$$
p_{\boldsymbol{\theta}}(\mathbf{x}) = \int p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) \, d\mathbf{z}
$$  

을 해석적으로 풀거나 효율적으로 추정할 방법이 없기 때문이다.  

따라서 이 주변우도(marginal likelihood) 또는 모델 증거(model evidence)를 계산할 수 없게 되며,  
그 결과 완전 관측 모델에서처럼  
모델 파라미터에 대해 미분(differentiation)하고 최적화(optimization)하는 것이 불가능해진다.

---

$p_{\boldsymbol{\theta}}(\mathbf{x})$의 계산 불가능성(intractability)은  
사후분포(posterior distribution) $p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})$의 계산 불가능성과 밀접한 관련이 있다.  

결합분포(joint distribution) $p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})$는 효율적으로 계산할 수 있지만,  
이 두 확률밀도는 다음의 기본 관계식으로 연결된다.

$$
p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})
= \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}
{p_{\boldsymbol{\theta}}(\mathbf{x})}
\tag{1.19}
$$

---

$p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})$는 계산 가능한(tractable) 분포이므로,  
만약 주변우도 $p_{\boldsymbol{\theta}}(\mathbf{x})$를 계산할 수 있다면 사후분포 $p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})$ 역시 계산할 수 있다.  
반대로, 사후분포를 계산할 수 있다면 주변우도 역시 계산할 수 있다.  

그러나 DLVM에서는 이 두 분포가 모두 계산 불가능(intractable)하다.

---

근사 추론 기법(approximate inference techniques, 부록 A.2 참조)은  
DLVM에서 사후분포 $p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})$와 주변우도 $p_{\boldsymbol{\theta}}(\mathbf{x})$를 근사적으로 계산할 수 있게 한다.  

전통적인 추론 방법들은 계산 비용이 상대적으로 높다.  

예를 들어, 이러한 방법들은  
각 데이터 포인트마다 별도의 최적화 루프(per-datapoint optimization loop)를 요구하거나,  
사후분포를 부정확하게 근사하는 경우가 많다.  

따라서 이러한 비효율적인 절차는 피하고자 한다.

---

마찬가지로, 신경망으로 파라미터화된 유향 모델(directed models)의  
모수(parameter)에 대한 사후분포 $p(\boldsymbol{\theta} \mid \mathcal{D})$ 역시  
정확하게 계산하기 어렵기 때문에(intractable), 근사 추론 기법이 필요하다.