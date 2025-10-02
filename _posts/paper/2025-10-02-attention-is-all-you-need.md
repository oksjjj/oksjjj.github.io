---
layout: post
title: "[논문] Attention Is All You Need"
date: 2025-10-02 12:40:00 +0900
categories:
  - "논문"
tags: []
---

# Attention Is All You Need  

**저자**  
Ashish Vaswani (Google Brain) - avaswani@google.com  
Noam Shazeer (Google Brain) - noam@google.com  
Niki Parmar (Google Research) - nikip@google.com  
Jakob Uszkoreit (Google Research) - usz@google.com  
Llion Jones (Google Research) - llion@google.com  
Aidan N. Gomez † (University of Toronto) - aidan@cs.toronto.edu  
Łukasz Kaiser (Google Brain) - lukaszkaiser@google.com  
Illia Polosukhin ‡ - illia.polosukhin@gmail.com  

---

> **주석**  
> ∗ 공동 기여(Equal contribution). 저자 순서는 무작위이다.  
> - Jakob은 RNN을 Self-Attention으로 대체하자는 아이디어를 제안하고, 이를 평가하기 위한 노력을 시작했다.  
> - Ashish는 Illia와 함께 최초의 Transformer 모델을 설계하고 구현했으며, 이 연구의 모든 측면에 핵심적으로 관여했다.  
> - Noam은 Scaled Dot-Product Attention, Multi-Head Attention, 학습해야 할 추가 파라미터 없이 정의되는 위치 표현(position representation)을 제안했으며, 연구의 거의 모든 세부 사항에 깊이 참여했다.  
> - Niki는 원래 코드베이스와 tensor2tensor에서 수많은 모델 변형을 설계, 구현, 튜닝, 평가하였다.  
> - Llion은 새로운 모델 변형을 실험했으며, 초기 코드베이스, 효율적인 추론 및 시각화를 담당했다.  
> - Lukasz와 Aidan은 tensor2tensor의 다양한 부분을 설계하고 구현하는 데 수많은 시간을 투자하여, 초기 코드베이스를 대체하고 결과를 크게 개선했으며 연구 속도를 비약적으로 가속시켰다.  
> 
> † Google Brain에서 수행한 연구 결과임.  
> ‡ Google Research에서 수행한 연구 결과임.  
> 
> *본 논문은 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA에서 발표되었다.*

---

## 초록 (Abstract)  

현재 널리 사용되는 **시퀀스 변환(sequence transduction) 모델**들은 인코더와 디코더를 포함하는 **복잡한 순환 신경망(recurrent neural networks)** 또는 **합성곱 신경망(convolutional neural networks)**을 기반으로 한다.  
가장 성능이 좋은 모델들은 또한 **어텐션 메커니즘(attention mechanism)**을 통해 인코더와 디코더를 연결한다.  

우리는 전적으로 **어텐션 메커니즘**에만 기반한 새로운 단순한 네트워크 아키텍처, 즉 **Transformer**를 제안한다. 이 아키텍처는 순환과 합성곱을 완전히 제거한다.  

두 가지 기계 번역 과제에서의 실험 결과, 제안한 모델은 **더 높은 품질**을 보이는 동시에 **병렬화(parallelization)**가 더 잘 가능하고, **훈련에 필요한 시간**도 크게 줄어든다는 것을 보여주었다.  

우리 모델은 **WMT 2014 영어→독일어 번역 과제**에서 **28.4 BLEU** 점수를 달성했으며, 이는 기존 최고 결과(앙상블 모델 포함)를 **2 BLEU 이상** 초과한 성과이다.  
또한 **WMT 2014 영어→프랑스어 번역 과제**에서, 단일 모델 기준으로 최고 성능을 경신하며, **BLEU 점수 41.0**을 달성했다. 이는 8개의 GPU에서 3.5일간 훈련한 결과이며, 문헌에 보고된 기존 최고 성능 모델들의 훈련 비용의 일부만으로 달성한 것이다.  
  
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
> BLEU 점수는 일반적으로 $0 \sim 1$ 범위를 가지며, **백분율(0~100점)**로 환산해 보고된다. 값이 높을수록 번역이 더 정확하고 자연스러움을 의미한다.

---

## 1 서론 (Introduction)  

순환 신경망(Recurrent Neural Networks, RNN), 장단기 메모리(Long Short-Term Memory, LSTM) [12], 그리고 게이트 순환 신경망(Gated Recurrent Neural Networks) [7]은 시퀀스 모델링과 변환(transduction) 문제(예: 언어 모델링, 기계 번역 [29, 2, 5])에서 확고하게 최첨단(state-of-the-art) 접근법으로 자리 잡아왔다. 이후 수많은 연구들이 계속해서, 순환 언어 모델과 인코더-디코더 아키텍처 [31, 21, 13]의 적용 범위를 확장하려는 노력을 이어왔다.  

순환 모델은 일반적으로 입력과 출력 시퀀스의 기호 위치(symbol positions)에 따라 연산을 단계적으로 진행한다. 각 위치들을 연산 시간의 단계(computation time steps)들과 정렬(align)시키기 위해, 은닉 상태 $h_t$를 생성하는데, 이는 이전 은닉 상태 $h_{t-1}$과 위치 $t$의 입력의 함수로 정의된다. 이러한 본질적인 순차적(sequential) 특성 때문에 학습 샘플들 내부에서의 병렬화가 불가능하다. 시퀀스 길이가 길어질수록 이 문제가 더 심각해지는데, 이는 GPU 메모리 제약으로 인해 샘플들 간에 동시에 처리할 수 있는 배치 크기가 제한되기 때문이다. 최근 연구에서는 팩터라이제이션(factorization) 트릭 [18]과 조건부 연산(conditional computation) [26]을 통해 계산 효율성을 크게 개선했고, 후자의 경우 모델 성능 향상도 이루어졌다. 그러나 순차적 계산이라는 근본적인 제약은 여전히 남아 있다.  

어텐션 메커니즘(attention mechanisms)은 다양한 과제에서 강력한 시퀀스 모델링 및 변환 모델의 핵심적인 요소가 되었다. 이는 입력 또는 출력 시퀀스에서 거리에 상관없이 의존성을 모델링할 수 있도록 한다 [2, 16]. 하지만 극히 일부의 경우 [22]를 제외하면, 이러한 어텐션 메커니즘은 항상 순환 신경망과 결합되어 사용되어 왔다.  

본 연구에서는 순환 구조를 제거하고 대신 입력과 출력 간의 전역적 의존성을 학습하기 위해 전적으로 어텐션 메커니즘에 의존하는 모델 아키텍처인 **Transformer**를 제안한다. Transformer는 훨씬 더 높은 수준의 병렬화를 가능하게 하며, 단 8개의 P100 GPU에서 12시간 정도의 훈련만으로도 번역 품질에서 새로운 최첨단 성능(state-of-the-art)에 도달할 수 있었다.  
