---
layout: post
title: "[논문] Attention Is All You Need"
date: 2025-10-02 12:40:00 +0900
categories:
  - "논문"
tags: []
---

> **논문 출처**  
> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.  
> *Attention Is All You Need*.  
> Advances in Neural Information Processing Systems (NeurIPS 2017).  
> <a href="https://arxiv.org/abs/1706.03762" target="_blank">🔗 원문 링크 (arXiv:1706.03762)</a>

# Attention Is All You Need  

**저자**  
- Ashish Vaswani (Google Brain) - avaswani@google.com  
- Noam Shazeer (Google Brain) - noam@google.com  
- Niki Parmar (Google Research) - nikip@google.com  
- Jakob Uszkoreit (Google Research) - usz@google.com  
- Llion Jones (Google Research) - llion@google.com  
- Aidan N. Gomez † (University of Toronto) - aidan@cs.toronto.edu  
- Łukasz Kaiser (Google Brain) - lukaszkaiser@google.com  
- Illia Polosukhin ‡ - illia.polosukhin@gmail.com  

---

**주석**  
  
∗ 공동 기여(Equal contribution). 저자 순서는 무작위이다.  
  
- Jakob은 RNN을 셀프-어텐션으로 대체하자는 아이디어를 제안하고, 이를 검증하기 위한 연구를 시작했다.  
- Ashish는, Illia와 함께 최초의 Transformer 모델을 설계하고 구현했으며, 이 연구의 모든 측면에 핵심적으로 관여했다.  
- Noam은 Scaled Dot-Product Attention, Multi-Head Attention, 학습 파라미터가 아닌 위치 표현(parameter-free position representation)을 제안했으며, 연구의 거의 모든 세부 사항에 깊이 참여했다.  
- Niki는 우리의 오리지널 코드베이스와 **tensor2tensor**에서 수많은 모델 변형을 설계, 구현, 튜닝, 평가하였다.  
- Llion은 새로운 모델 변형을 실험했을 뿐만 아니라, 초기 코드베이스와 효율적인 추론 및 시각화를 맡았다.  
- Lukasz와 Aidan은 tensor2tensor의 다양한 부분을 설계하고 구현하는 데 수많은 시간을 투자하여, 초기 코드베이스를 대체하고 결과를 크게 개선했으며 연구 속도를 비약적으로 가속시켰다.  
 
† Google Brain에서 수행한 연구 결과임.  
‡ Google Research에서 수행한 연구 결과임.  
 
*본 논문은 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA에서 발표되었다.*. 
  
---

> **(블로그 추가 설명) Tensor2Tensor란?**  
> Tensor2Tensor(T2T)는 구글 브레인 팀이 공개한 **오픈소스 딥러닝 라이브러리**로,  
> 주로 **시퀀스-투-시퀀스(Sequence-to-Sequence)** 작업(기계 번역, 요약, 언어 모델링 등)을 쉽게 실험할 수 있도록 설계되었다.  
> 
> 주요 특징:  
> - **TensorFlow 기반**으로 구현됨  
> - 다양한 시퀀스 모델(RNN, CNN, Transformer 등)과 대규모 데이터셋을 내장  
> - GPU/TPU 병렬 학습 지원으로 대규모 실험이 용이  
> 
> Transformer 논문 역시 초기에는 Tensor2Tensor 코드베이스 안에서 개발되고 검증되었으며,  
> 이 때문에 저자 기여 부분에서 여러 연구자들이 **tensor2tensor의 설계·구현·개선**을 언급하고 있다.  

---

## 초록 (Abstract)  

현재 널리 사용되는 시퀀스 변환(sequence transduction) 모델들은  
인코더와 디코더를 포함하는 복잡한 순환 신경망(recurrent neural networks) 또는  
합성곱 신경망(convolutional neural networks)을 기반으로 한다.  

가장 성능이 좋은 모델들은 **어텐션 메커니즘(attention mechanism)**을 통해  
인코더와 디코더를 연결한다.  

우리는 전적으로 어텐션 메커니즘에만 기반한 새로운 단순한 네트워크 아키텍처,  
즉 **Transformer**를 제안한다. 이 아키텍처는 순환과 합성곱을 완전히 제거했다.  

두 가지 기계 번역 과제에서의 실험 결과, 제안한 모델은 더 높은 품질을 보이는 동시에  
병렬화가 용이하며, 훈련에 필요한 시간도 크게 줄어든다는 것을 보여주었다.  

우리 모델은 WMT 2014 영어→독일어 번역 과제에서 28.4 **BLEU** 점수를 달성했으며,  
이는 이전 최고 성능(앙상블 모델 포함)을 2 BLEU 이상 능가한 결과이다.  

또한 WMT 2014 영어→프랑스어 번역 과제에서, 단일 모델 기준으로 최고 성능을 경신하며  
BLEU 점수 41.0을 달성했다. 이는 8개의 GPU에서 3.5일간 훈련한 결과이며,  
문헌에 보고된 기존 최고 성능 모델들의 훈련 비용의 일부만으로 달성한 것이다.  
  
---  

> **(블로그 추가 설명) BLEU 점수란?**  
> BLEU(Bilingual Evaluation Understudy)는 기계 번역의 품질을 평가하는 대표적인 지표이다.  
> 예측 번역문과 정답 번역문 사이의 **N-gram 일치율(precision)**을 측정하여 계산한다.  
> 
> BLEU의 정의 수식은 다음과 같다:  
> 
> $$
> BLEU = BP \cdot \left( \prod_{i=1}^4 \text{precision}_i \right)^{\tfrac{1}{4}}
> $$  
> 
> - $\text{precision}_i$: 예측 번역문과 정답 번역문 간의 $i$-gram precision  
> - $BP$: **brevity penalty**, 짧은 문장이 과도하게 높은 점수를 받는 것을 막기 위한 보정값  
>   $$
>   BP = \min \left( 1, \frac{\text{예측 문장의 단어 수}}{\text{정답 문장의 단어 수}} \right)
>   $$  
> 
> BLEU 점수는 일반적으로 $0 \sim 1$ 범위를 가지며, **백분율(0~100점)**로 환산해 보고된다.  
  값이 높을수록 번역이 더 정확하고 자연스러움을 의미한다.

---

## 1 서론 (Introduction)  

순환 신경망(Recurrent Neural Networks, RNN), 장단기 메모리(Long Short-Term Memory, LSTM) [12],  
그리고 게이트 순환 신경망(Gated Recurrent Neural Networks) [7]은  
언어 모델링, 기계 번역 [29, 2, 5] 등과 같은 시퀀스 모델링과 변환(transduction) 문제에서  
확고하게 최첨단(state-of-the-art) 접근법으로 자리 잡아왔다.  

이후 언어 모델과 인코더-디코더 아키텍처의 적용 범위를 확장하기 위한 수많은 연구가 이어졌다 [31, 21, 13].  

---

순환 모델은 일반적으로 입력과 출력 시퀀스의 기호 위치를 기준으로 연산을 나누어 수행한다.  
입력 시퀀스의 각 위치를 연산 시간 단계에 대응시켜, 일련의 은닉 상태 $h_t$ 들을 만들어낸다.  

여기서 $h_t$는 이전 은닉 상태 $h_{t-1}$와 **위치 $t$의 입력**의 함수이다.  

이러한 본질적인 순차적(sequential) 특성 때문에 하나의 학습 샘플 내에서는 병렬화가 불가능하다.  
시퀀스 길이가 길어질수록 이 문제가 더 심각해지는데,  
GPU 메모리 제약으로 인해 샘플들 간에 동시에 처리할 수 있는 배치 크기가 제한되기 때문이다.  

최근 연구에서는 **팩터라이제이션(factorization) 트릭** [18]과 **조건부 연산(conditional computation)** [26]을 통해  
계산 효율성이 크게 향상되었으며, 조건부 연산에서는 모델 성능 또한 개선되었다.  

그러나 순차적 계산이라는 근본적인 제약은 여전히 남아 있다.  

---

> **(블로그 추가 설명) 팩터라이제이션 트릭(Factorization Tricks)**  
> 순환 신경망(RNN)이나 언어 모델에서 등장하는 거대한 확률 분포(예: 소프트맥스 출력층)를 직접 계산하면 연산량이 매우 커진다.  
> 팩터라이제이션 트릭은 이러한 연산을 **작은 부분으로 분해(factorize)** 하여 효율적으로 계산하는 기법을 의미한다.  
> 대표적으로 **Hierarchical Softmax**, **Sampled Softmax**, **Noise-Contrastive Estimation(NCE)** 등이 이에 속한다.  
> 즉, 동일한 모델을 더 적은 계산량으로 학습할 수 있도록 하는 최적화 기법이다.  

---

> **(블로그 추가 설명) 조건부 연산(Conditional Computation)**  
> 신경망의 모든 파라미터를 항상 사용하는 대신, **입력에 따라 일부 연산만 선택적으로 활성화**하는 방식을 말한다.  
> 예를 들어, 특정 입력에서는 네트워크의 일부 경로만 계산하여 연산량을 줄이고,  
> 중요한 입력에 대해서는 더 많은 계산을 할당할 수 있다.  
> 이는 **효율성을 높이는 동시에 모델의 표현력도 유지하거나 향상**시킬 수 있어, 최근 대규모 신경망 연구에서 주목받는 방법이다.  
> 
> 수식으로 표현하면, 전체 함수 $f(x)$ 대신에  
> $$
> f(x) = \sum_i g_i(x) f_i(x)
> $$  
> 와 같이, 입력 $x$에 따라 선택 함수 $g_i(x)$가 특정 모듈 $f_i$만 활성화되도록 설계한다.  

---

어텐션 메커니즘(attention mechanisms)은 다양한 과제에서  
강력한 시퀀스 모델링 및 변환 모델의 핵심적인 요소가 되었다.  

이는 입력 또는 출력 시퀀스에서 거리에 상관없이 의존성을 모델링할 수 있도록 한다. [2, 16]  
하지만 극히 일부의 경우 [22]를 제외하면, 이러한 어텐션 메커니즘은  
항상 순환 신경망과 결합되어 사용되어 왔다.  

본 연구에서는 순환 구조를 제거하고, 대신 입력과 출력 간의 전역적 의존성을 학습하기 위해  
전적으로 어텐션 메커니즘에 의존하는 모델 아키텍처인 Transformer를 제안한다.  

Transformer는 훨씬 더 높은 수준의 병렬화를 가능하게 하며,  
단 8개의 P100 GPU에서 12시간 정도의 훈련만으로도  
번역 품질에서 새로운 최첨단 성능(state-of-the-art)에 도달할 수 있었다.  

---
  
## 2 배경 (Background)  
  
순차적 계산을 줄이려는 목표는 **Extended Neural GPU** [20], **ByteNet** [15], **ConvS2S** [8]의 기반이 되기도 한다.  
이들 모델은 모두 합성곱 신경망(convolutional neural networks)을 기본 구성 요소로 사용하여,  
입력과 출력의 모든 위치에 대한 은닉 표현(hidden representations)을 병렬적으로 계산한다.

이들 모델에서는 두 임의의 입력 또는 출력 위치 간 신호를 연결하기 위해 필요한 연산의 수가  
위치 간 거리(distance)에 따라 증가한다.  
ConvS2S의 경우 선형적으로(linearly), ByteNet의 경우 로그(logarithmically) 비율로 증가한다.

이로 인해 먼 위치들 사이의 의존성을 학습하는 것이 더욱 어려워진다 [11].  

Transformer에서는 이러한 연산이 **상수 개수의 연산**으로 줄어든다.  
다만 어텐션 가중치가 적용된 위치들을 평균하기 때문에 **표현의 세밀함(정밀도)**이 줄어드는 비용이 따른다.  
우리는 이를 3.2절에서 설명하는 **멀티-헤드 어텐션(Multi-Head Attention)**으로 보완한다.

---

> **(블로그 추가 설명) Extended Neural GPU**  
> Extended Neural GPU는 2015년 발표된 모델로, **게이트드 순환 유닛(GRU)**와 **합성곱(convolution)**을 결합한 아키텍처이다.  
> 이 모델은 입력 시퀀스를 2차원 격자 구조로 표현하고, 합성곱 연산을 통해 병렬적으로 처리함으로써  
> 전통적인 RNN보다 더 효율적으로 긴 시퀀스를 학습할 수 있도록 설계되었다.  
> Transformer로 가는 중간 단계의 아이디어 중 하나로 평가된다.  
> 
> ---

> **(블로그 추가 설명) ByteNet**  
> ByteNet은 2016년 발표된 모델로, **합성곱 신경망(CNN)** 기반의 시퀀스-투-시퀀스 아키텍처이다.  
> 입력 길이에 따라 연산량이 선형적으로 늘어나는 RNN과 달리,  
> ByteNet은 **직접적인 병렬 처리**와 **지수적 크기의 수용영역(receptive field)**을 활용하여  
> 긴 시퀀스에서도 효율적으로 문맥을 포착할 수 있다.  
> 다만 두 위치 간 의존성을 학습하는 데 필요한 연산이 거리의 로그(logarithmic)에 비례해 증가하는 제약이 있다.  
> 
> ---

> **(블로그 추가 설명) ConvS2S**  
> ConvS2S(Convolutional Sequence to Sequence)는 2017년 발표된 모델로,  
> **완전히 합성곱 네트워크만으로 인코더-디코더 구조**를 구성한 기계 번역 모델이다.  
> 합성곱 계층을 깊게 쌓아 **넓은 문맥 범위**를 처리할 수 있으며,  
> RNN보다 **병렬화가 용이하고 학습 속도가 빠르다**는 장점을 가진다.  
> 그러나 입력 위치 간 거리에 비례해 연산량이 선형적으로(linearly) 증가한다는 한계가 있다.  

---

셀프-어텐션(Self-attention)은 때로 인트라-어텐션(intra-attention)이라고도 불리며,  
하나의 시퀀스 내 서로 다른 위치들을 연결하여 그 시퀀스의 표현을 계산하는 어텐션 메커니즘이다.

셀프-어텐션은 독해(reading comprehension), 추상적 요약(abstractive summarization),  
텍스트 함의(textual entailment), 과제와 무관한 문장 표현 학습(task-independent sentence representations) [4, 22, 23, 19] 등  
다양한 작업에서 성공적으로 활용되어 왔다.

**엔드-투-엔드 메모리 네트워크(End-to-end memory networks)**는  
시퀀스 정렬 기반 순환(sequence-aligned recurrence) 대신 **순환 어텐션 메커니즘(recurrent attention mechanism)**에 기반하며,  
단순 언어 질의응답(simple-language question answering)과 언어 모델링(language modeling) 과제에서  
우수한 성능을 보인 것으로 보고되었다 [28].
  
그러나 우리가 아는 한, Transformer는 입력과 출력의 표현을 계산함에 있어  
시퀀스 정렬 기반 RNN이나 합성곱을 사용하지 않고 전적으로 **셀프-어텐션(Self-attention)**에만 의존하는  
최초의 변환(transduction) 모델이다.

다음 섹션들에서 우리는 Transformer를 설명하고,  
셀프-어텐션(Self-attention)의 필요성을 제시하며,  
[14, 15], [8]과 같은 모델들에 비해 가지는 장점을 논의할 것이다.

---

> **(블로그 추가 설명) 엔드-투-엔드 메모리 네트워크(End-to-End Memory Networks)**  
> 엔드-투-엔드 메모리 네트워크는 2015년 Facebook AI Research에서 제안한 모델로,  
> **메모리(memory) 컴포넌트**와 **어텐션(attention) 메커니즘**을 결합하여,  
> 순환 신경망(RNN)의 시퀀스 정렬 기반 연산 대신 **순환 어텐션(recurrent attention)**을 사용한다.  
> 
> 주요 아이디어는 모델이 **외부 메모리**를 참조하며 필요한 정보를 읽고 쓸 수 있도록 하여,  
> 긴 문맥에서의 추론(reasoning)을 가능하게 한다는 것이다.  
> 
> - **순환 어텐션(recurrent attention)**이란?  
>   한 번의 어텐션으로 끝나는 것이 아니라, 여러 번 반복적으로(attentive hops)  
>   메모리에 주의를 기울여 필요한 정보를 단계적으로 모으는 방식이다.  
>   → 예: 질문에 답하기 위해 먼저 인물 정보를 찾고, 다음에 사건 정보를 찾는 식으로 차례대로 집중.  
> 
> 대표적인 응용 분야:  
> - **질의응답(Question Answering, QA)** : 긴 문서 속에서 답변에 필요한 정보를 단계적으로 추출  
> - **언어 모델링(Language Modeling)** : 문맥을 메모리에 저장·활용하여 다음 단어를 더 잘 예측  
> 
> 연구 결과, 메모리 네트워크는 단순 언어 질의응답(Simple QA)과 언어 모델링 과제에서  
> 우수한 성능을 보인 것으로 보고되었다 [28].  

---

## 3 모델 아키텍처 (Model Architecture)

대부분의 경쟁력 있는 신경망 기반 시퀀스 변환 모델들은 인코더-디코더 구조를 가진다 [5, 2, 29].

여기에서 인코더는 기호 표현들 $(x_1, \ldots, x_n)$로 이루어진 입력 시퀀스를  
연속적 표현들의 시퀀스 $z = (z_1, \ldots, z_n)$로 매핑한다.

$z$가 주어지면, 디코더는 기호들로 이루어진 출력 시퀀스 $(y_1, \ldots, y_m)$을  
한 번에 하나씩 순차적으로 생성한다.

각 단계에서 모델은 **자기회귀적(auto-regressive)** [9]으로 동작하며,  
다음 출력을 생성할 때 이전에 생성된 기호들을 추가 입력으로 사용한다.

Transformer는 전체적으로 이러한 아키텍처를 따르며,  
인코더와 디코더 모두에서 **셀프-어텐션(Self-attention)**과 **위치별(point-wise) 완전 연결 계층(fully connected layers)**을 적층하여 사용한다.  
이는 각각 그림 1의 왼쪽과 오른쪽 부분에 나타나 있다.

---

**그림 1. Transformer - 모델 아키텍처**

<img src="/assets/img/paper/attention-is-all-you-need/image_1.png" alt="image" width="480px"> 

---

### 3.1 인코더와 디코더 스택 (Encoder and Decoder Stacks)  

---

**인코더(Encoder)**:  

인코더는 $N = 6$개의 동일한 층(layer)으로 이루어진 스택으로 구성된다.

각 층은 두 개의 하위 층(sub-layer)으로 구성된다.

첫 번째 하위 층은 **멀티-헤드 셀프-어텐션(Multi-Head Self-Attention) 메커니즘**이고,  
두 번째 하위 층은 단순한 **위치별(position-wise) 완전 연결 피드포워드 네트워크**이다.

각 두 개의 하위 층(sub-layer)에는 **잔차 연결(residual connection)** [10]을 적용하고,  
그 뒤에 **레이어 정규화(layer normalization)** [1]를 수행한다.

즉, 각 하위 층(sub-layer)의 출력은 다음과 같다.  

---

$$
\text{LayerNorm}(x + \text{Sublayer}(x))
$$

---

여기서 $\text{Sublayer}(x)$는 해당 하위 층이 구현하는 함수이다.

이러한 잔차 연결(residual connections)을 가능하게 하기 위해,  
모델의 모든 하위 층(sub-layer)과 임베딩 층(embedding layer)은  
차원 $d_{\text{model}} = 512$의 출력을 생성한다.  

---

**디코더(Decoder)**:  

디코더 역시 $N = 6$개의 동일한 층(layer)으로 이루어진 스택으로 구성된다.

각 인코더 층에 포함된 두 개의 하위 층(sub-layer)에 더해,  
디코더는 세 번째 하위 층을 추가하는데,  
이 층은 인코더 스택의 출력을 대상으로 **멀티-헤드 어텐션(Multi-Head Attention)**을 수행한다.

인코더와 마찬가지로, 디코더의 각 하위 층(sub-layer)에도  
**잔차 연결(residual connection)**을 적용하고, 그 뒤에 **레이어 정규화(layer normalization)**를 수행한다.

또한 디코더 스택의 셀프-어텐션(Self-Attention) 하위 층을 수정하여,  
각 위치가 이후의 위치들에 주의를 기울이지 못하도록(참조하지 못하도록) 한다.

이러한 마스킹(masking)은 출력 임베딩(output embedding)이 한 위치만큼 어긋나 있다는 사실과 결합되어,  
위치 $i$에서의 예측이 $i$보다 작은 위치들의 알려진 출력에만 의존하도록 보장한다.  

---

### 3.2 어텐션 (Attention)

어텐션 함수(attention function)는 **쿼리(query)**와 **키-값(key-value) 쌍들의 집합**을 출력으로 매핑하는 함수로 설명할 수 있으며,  
이때 쿼리, 키, 값, 출력은 모두 벡터(vector)이다.

출력은 값(value)들의 가중합(weighted sum)으로 계산되며,  
각 값에 할당되는 가중치는 쿼리(query)와 해당 키(key) 간의 **호환 함수(compatibility function)**에 의해 계산된다.  

---

#### 3.2.1 스케일드 닷-프로덕트 어텐션 (Scaled Dot-Product Attention)

우리는 우리가 사용하는 특정한 어텐션 방식을 **“스케일드 닷-프로덕트 어텐션(Scaled Dot-Product Attention)”**이라고 부른다 (그림 2).

---

**그림 2.** (왼쪽) 스케일드 닷-프로덕트 어텐션(Scaled Dot-Product Attention).  
(오른쪽) 멀티-헤드 어텐션(Multi-Head Attention)은 여러 개의 어텐션 층을 병렬로 수행한다.

<img src="/assets/img/paper/attention-is-all-you-need/image_2.png" alt="image" width="720px"> 

---

입력은 차원 $d_k$의 **쿼리(query)**와 **키(key)**, 그리고 차원 $d_v$의 **값(value)**으로 구성된다.

쿼리를 모든 키들과 내적(dot product)한 뒤,  
각 결과를 $\sqrt{d_k}$로 나누고, 소프트맥스(softmax) 함수를 적용하여  
값(value)들에 대한 가중치(weights)를 얻는다.

실제로는 여러 개의 쿼리(query) 집합에 대해 어텐션 함수를 동시에 계산하며,  
이 쿼리들은 함께 묶여 행렬 $Q$로 표현된다.

키(keys)와 값(values) 또한 각각 행렬 $K$와 $V$로 함께 묶어 표현된다.

출력 행렬은 다음과 같이 계산된다:  

---

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V \tag{1}
$$

---

가장 흔히 사용되는 두 가지 어텐션 함수는  
**가산적 어텐션(additive attention)** [2]과  
**내적(곱셈적) 어텐션(dot-product / multiplicative attention)**이다.

내적 어텐션(dot-product attention)은  
스케일링 계수 $\tfrac{1}{\sqrt{d_k}}$를 제외하면 우리의 알고리즘과 동일하다.

가산적 어텐션(additive attention)은  
하나의 은닉층(hidden layer)을 가진 피드포워드 네트워크를 사용하여  
호환 함수(compatibility function)를 계산한다.

이 두 방법은 이론적인 복잡도 면에서는 유사하지만,  
내적 어텐션(dot-product attention)은 고도로 최적화된 행렬 곱셈(matrix multiplication) 코드로 구현할 수 있기 때문에  
실제로는 훨씬 더 빠르고 공간 효율적이다.

$d_k$의 값이 작은 경우에는 두 메커니즘이 비슷한 성능을 보이지만,  
$d_k$가 큰 경우에는 스케일링을 적용하지 않은 내적 어텐션(dot-product attention)보다  
가산적 어텐션(additive attention)이 더 나은 성능을 보인다 [3].

우리는 $d_k$가 큰 경우, 내적(dot product) 값의 크기가 커져  
소프트맥스(softmax) 함수가 기울기(gradient)가 극도로 작은 영역에 놓이게 된다고 추측한다 [4].

이러한 효과를 상쇄하기 위해, 우리는 내적(dot product)을 $\tfrac{1}{\sqrt{d_k}}$로 스케일링한다.

---

**주석**  

[4] 왜 내적(dot product)이 커지는지를 설명하기 위해,  
$q$와 $k$의 성분들이 평균 0, 분산 1을 갖는 서로 독립인 확률 변수라고 가정하자.  
그렇다면 내적 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$는 평균 0, 분산 $d_k$를 갖게 된다.  

---

#### 3.2.2 멀티-헤드 어텐션 (Multi-Head Attention)

$d_{model}$ 차원의 키(key), 값(value), 쿼리(query)에 대해 단일 어텐션 함수를 수행하는 대신,  
우리는 쿼리, 키, 값을 서로 다른 학습된 선형 변환(linear projection)을 통해  
각각 $h$번 선형 투영하여, 차원 $d_k$, $d_k$, $d_v$로 매핑하는 것이 유익하다는 것을 발견했다.

이렇게 투영된 각각의 쿼리(query), 키(key), 값(value)에 대해  
우리는 어텐션 함수를 병렬로 수행하며,  
그 결과 $d_v$ 차원의 출력 값(output value)을 얻게 된다.

이 출력들은 합쳐진 후(concatenate), 다시 한 번 선형 투영(linear projection)되어  
최종 출력 값들을 생성하게 되며, 이는 그림 2에 나타나 있다.

멀티-헤드 어텐션(Multi-Head Attention)은  
모델이 서로 다른 표현 부분공간(representation subspace)들에서,  
서로 다른 위치들의 정보를 동시에 주목(attend)할 수 있도록 한다.

하나의 어텐션 헤드만 사용하는 경우, 평균화(averaging)가 이러한 능력을 방해하게 된다.  
  
---
  
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(Q W_i^Q, \; K W_i^K, \; V W_i^V)
$$  
  
여기서 투영(projection)들은 파라미터 행렬로, 다음과 같이 정의된다:

$$
W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad
W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad
W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v},
$$

$$
W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}.
$$

---

이 연구에서 우리는 $h = 8$개의 병렬 어텐션 층(layer), 즉 헤드(head)를 사용한다.

각 헤드마다 우리는 $d_k = d_v = d_{\text{model}} / h = 64$를 사용한다.

각 헤드의 차원이 줄어들기 때문에,  
전체 계산 비용은 전체 차원을 사용하는 단일 헤드 어텐션과 유사하다.

---

#### 3.2.3 본 모델에서의 어텐션 활용 (Applications of Attention in our Model)

Transformer는 멀티-헤드 어텐션(Multi-Head Attention)을 세 가지 방식으로 사용한다:

- **인코더-디코더 어텐션(encoder-decoder attention)** 층에서는,  
  쿼리(query)는 전 단계의 디코더 층에서 나오며,  
  키(key)와 값(value)은 인코더의 출력으로부터 나온다.  
  이를 통해 디코더의 각 위치는 입력 시퀀스의 모든 위치에 주의를 기울일 수 있게 된다.  
  이는 [31, 2, 8]과 같은 시퀀스-투-시퀀스(sequence-to-sequence) 모델에서의  
  전형적인 인코더-디코더 어텐션 메커니즘을 모방한 것이다.

- **인코더(encoder)**는 셀프-어텐션(self-attention) 층을 포함한다.  
  셀프-어텐션(self-attention) 층에서는 키(key), 값(value), 쿼리(query)가 모두 같은 곳에서 나오는데,  
  이 경우 그것은 인코더의 전 단계 층의 출력이다.  
  인코더의 각 위치는 인코더의 전 단계 층에 있는 모든 위치에 주의를 기울일 수 있다.  

- 비슷하게, **디코더(decoder)**의 셀프-어텐션(self-attention) 층에서는  
  디코더의 각 위치가 그 위치를 포함하여, 그 이전까지의 모든 위치에 주의를 기울일 수 있다.  
  자기회귀(auto-regressive) 특성을 유지하기 위해,  
  디코더에서는 미래 위치(오른쪽)에서 과거 위치(왼쪽)로의 정보 흐름(**leftward information flow**)을 차단해야 한다.  
  우리는 이러한 동작을 스케일드 닷-프로덕트 어텐션(scaled dot-product attention) 내부에서 구현한다.  
  즉, 소프트맥스(softmax) 입력에서 **허용되지 않는 연결(illegal connections)**에 해당하는 모든 값을  
  마스킹(masking)하여 $-\infty$로 설정한다. (그림 2 참조)

---

### 3.3 위치별 피드포워드 네트워크 (Position-wise Feed-Forward Networks)

어텐션 하위 층(sub-layer) 외에도,  
우리의 인코더와 디코더의 각 층은 완전 연결 피드포워드 네트워크(fully connected feed-forward network)를 포함한다.  
이 네트워크는 각 위치에 대해 **별도로(separately)**, 그리고 **동일하게(identically)** 적용된다.

이는 두 개의 선형 변환(linear transformation)으로 이루어져 있으며,  
그 사이에는 ReLU 활성화 함수(activation function)가 위치한다.

---

$$
\text{FFN}(x) = \max(0, \; xW_1 + b_1)W_2 + b_2 \tag{2}
$$

---

선형 변환(linear transformation)은 서로 다른 위치들에 대해서는 동일하게 적용되지만,  
층(layer)이 달라지면 서로 다른 파라미터(parameters)를 사용한다.

이를 다른 방식으로 표현하면, 커널 크기 1의 두 개의 합성곱(convolution)으로 볼 수 있다.

입력과 출력의 차원은 $d_{\text{model}} = 512$이고,  
내부 층(inner-layer)의 차원은 $d_{ff} = 2048$이다.

---

> **(블로그 추가 설명) FFN과 Conv1D(k=1)의 관계**  
> 위치별 피드포워드 네트워크(FFN)는 각 위치에 동일한 선형 변환(linear transformation)을 적용한다.  
> 이는 사실상 **커널 크기가 1인 1차원 합성곱(Conv1D with kernel size = 1)**과 동일하다.  
> 
> - Conv1D(k=1)은 커널 사이즈가 1이므로, **각각의 위치에 대해 동일한 가중치를 공유하는 필터**를 적용한다.  
> - 따라서 각 위치별로 독립적으로 연산되지만, 파라미터는 전체 시퀀스에서 공유된다.  
> - FFN 역시 입력 시퀀스의 각 위치마다 동일한 가중치 행렬을 적용하므로 동등하게 볼 수 있다.  
> 
> 따라서 구현 관점에서는 FFN을 Conv1D(k=1) 두 층을 쌓은 구조로 이해할 수 있다.

---

### 3.4 임베딩과 소프트맥스 (Embeddings and Softmax)

다른 시퀀스 변환(sequence transduction) 모델들과 유사하게,  
우리는 **학습된 임베딩(learned embedding)**을 사용하여  
입력 토큰(input token)과 출력 토큰(output token)을 $d_{\text{model}}$ 차원의 벡터로 변환한다.

또한 우리는 일반적으로 사용되는 **학습된 선형 변환(learned linear transformation)**과  
**소프트맥스(softmax) 함수**를 사용하여, 디코더의 출력을  
다음 토큰(next-token)의 예측 확률로 변환한다.

우리의 모델에서는 [24]와 유사하게,  
두 개의 임베딩 층(embedding layer)과 소프트맥스(softmax) 이전의 선형 변환(pre-softmax linear transformation)이  
**동일한 가중치 행렬(weight matrix)을 공유한다.**

임베딩 층(embedding layer)에서는,  
그 가중치(weight)에 $\sqrt{d_{\text{model}}}$을 곱한다.

---

> **(블로그 추가 설명) 임베딩과 출력층의 가중치 공유 (Weight Tying)**  
> 
> 일반적인 신경망 언어 모델에서는  
> - **입력 측(Input Embedding):** 단어의 원-핫 벡터(one-hot vector, 차원 $|V|$)를  
>   $d_{\text{model}}$ 차원의 임베딩 벡터로 변환한다.  
>   이때 사용하는 행렬은 $W \in \mathbb{R}^{|V| \times d_{\text{model}}}$ 이다.  
> 
> - **출력 측(Output Projection):** 디코더의 은닉 벡터($d_{\text{model}}$ 차원)를  
>   어휘 분포(vocabulary distribution, 차원 $|V|$)로 변환한다.  
>   보통 새로운 파라미터 $W' \in \mathbb{R}^{d_{\text{model}} \times |V|}$를 두지만,  
>   Transformer에서는 $W' = W^\top$로 두어 입력 임베딩과 동일한 행렬을 전치(transpose)하여 사용한다.  
>
> 이렇게 하면  
> 1. **파라미터 수가 줄어들어 효율성**이 좋아지고,  
> 2. **입출력 임베딩 공간을 일관성 있게 공유**하므로 학습이 안정된다.  
> 
> 이 기법은 *Press & Wolf (2017) [24]*에서 제안되었으며,  
> 이후 Transformer를 포함한 최신 언어 모델에서 표준적으로 채택되었다.

---

> **(블로그 추가 설명) 왜 $\sqrt{d_{\text{model}}}$을 곱하는가?**  
> 
> 임베딩 행렬의 각 원소는 평균 0, 분산 1로 초기화된다.  
> 이 경우, $d_{\text{model}}$ 차원 임베딩 벡터의 기대되는 크기(norm)는  
> 대략 $\sqrt{d_{\text{model}}}$ 정도가 된다.  
> 
> 그러나 $\sqrt{d_{\text{model}}}$은 상대적으로 작은 값이기 때문에,  
> 초기 학습 시 어텐션의 softmax가 거의 균등 분포를 내뱉는 문제가 생길 수 있다.  
> 
> 따라서 Transformer에서는 임베딩에 $\sqrt{d_{\text{model}}}$을 곱하여  
> norm 크기를 $\approx d_{\text{model}}$ 수준으로 맞춘다.  
> 이렇게 하면 임베딩 벡터의 스케일이 커져 학습이 안정되고,  
> softmax가 유의미한 확률 분포를 산출할 수 있다.  
> 
> 정리하면:  
> - **곱하기 전:** norm ≈ $\sqrt{d_{\text{model}}}$ (너무 작음)  
> - **곱한 후:** norm ≈ $d_{\text{model}}$ (적정한 크기)  
> 
> 반대로, 어텐션 내적에서는 값이 지나치게 커지므로  
> $\sqrt{d_k}$으로 나누어 균형을 맞춘다.  
> 즉, **임베딩에서는 곱하고, 어텐션에서는 나눈다.**

---