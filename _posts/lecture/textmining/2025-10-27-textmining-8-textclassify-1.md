---
layout: post
title: "[텍스트 마이닝] 8. Text Classification 1"
date: 2025-10-27 11:00:00 +0900
categories:
  - "대학원 수업"
  - "텍스트 마이닝"
tags: []
---

> 출처: 텍스트 마이닝 – 강성구 교수님, 고려대학교 (2025)

## p9. 텍스트 분류 (Text classification)

- 왜 중요한가?  
  - 데이터 분석의 핵심 과제: **미리 정의된 레이블을 할당(assign predefined labels)** (예: 브랜드, 감정, 주제 등)  
  - 검색 엔진, 추천 시스템, 스팸 탐지 등에서 폭넓게 사용된다.  
  - QA, 대화 시스템, 개인화(personalization) 등 다양한 고급 응용의 기초가 된다.  

<img src="/assets/img/lecture/textmining/8/image_1.png" alt="image" width="800px">

- 우리의 학습 경로:  
  - 텍스트를 벡터로 표현하였다.  
  - 다음으로, 이러한 표현을 이용하여 **분류(classification)** 를 수행할 것이다.  
  - 주로 **사전학습(pretrain) - 미세조정(fine-tune) 패러다임** 을 따라, 일반적인 언어 지식을 특정 분류 과제에 맞게 조정할 것이다.  

---

# p10. 사전학습 및 미세조정

---

## p11. 사전학습 + 미세조정 (Pretraining + Fine-tuning)

- 사전학습(Pretraining)  
  - 동기(Motivation): 웹에는 언어적 패턴과 세계 지식이 풍부하게 담긴 방대한 양의 텍스트 데이터가 존재한다.  
  - 목표: 사람의 레이블링 없이 **일반적인 목적의 표현(general-purpose representations)** 을 학습하는 것.  
  - 예시:  
    - Word2Vec (인접 단어 예측)  
    - BERT (마스크된 언어 모델링, 다음 문장 예측)  

<img src="/assets/img/lecture/textmining/8/image_2.png" alt="image" width="800px">

---

## p12. 사전학습 + 미세조정

- 미세조정(Fine-tuning)  
  - 동기(Motivation): 사전학습된 모델은 일반적인 지식을 포착하지만, 분류와 같은 작업에는 과제(task) 특화 지식이 필요하다.  
  - 목표: **사전학습된 모델의 파라미터를 조정(adjust the pretrained model parameters)** 하여 **특정 목표(downstream) 과제** 에서 좋은 성능을 내도록 하는 것.  
    - 처음부터 학습하는 것보다 훨씬 적은 레이블된 데이터(labeled data)만 필요하다.  
    - 사전학습 과정에서 학습된 지식을 활용한다.  
  - 예시:  
    - BERT를 이용한 감정 분류(sentiment classification)  

<img src="/assets/img/lecture/textmining/8/image_3.png" alt="image" width="800px">

---

## p13. 사전학습 + 미세조정

- 왜 사전학습 + 미세조정이 “최적화 관점(optimization perspective)”에서 도움이 되는가?  

- 사전학습(Pretraining)  
  - 손실함수 $L_{\text{pretrain}}$ 을 최소화하여 파라미터 $\hat{\theta}$ 를 학습한다.  
  - 좋은 초기값(good initialization)을 제공한다.  

- 미세조정(Fine-tuning)  
  - $\hat{\theta}$ 에서 시작하여 손실함수 $L_{\text{fine-tun}}$ 을 최소화한다.  
  - 사전학습된 모델을 목표 과제(target task)에 맞게 적응시킨다.  

- 확률적 경사 하강법(SGD)은 초기값(initialization)에 크게 영향을 받는다.  
- 사전학습으로부터 좋은 시작점을 얻으면,  
  모델은 효율적으로 수렴(converge efficiently)하며  
  더 적은 양의 데이터로도 잘 일반화(generalize well)되는 경향이 있다.  

<img src="/assets/img/lecture/textmining/8/image_4.png" alt="image" width="600px">

---

## p14. 사전학습 + 미세조정 예시

- Downstream task: 감정 분류  
  - 주어진 문장이 긍정(positive), 중립(neutral), 부정(negative)인지 예측한다.  

- 작동 방식:  
  - 사전학습된 BERT를 이용해 각 문장을 벡터로 표현한다.  
    - 일반적인 선택: CLS 표현(Representation) 또는 모든 문맥 임베딩의 평균.  
  - 작은 분류 헤드(classification head, 작은 신경망)를 추가한다.  
  - 이후 모델을 레이블이 있는 데이터(labeled data)로 미세조정(fine-tuning)한다.  

- 학습 선택:  
  A. 전체 미세조정(Full fine-tuning): 모든 파라미터(all parameters)를 업데이트한다.  
  B. 부분 미세조정(Partial fine-tuning): 일부 파라미터(subset of parameters)만 업데이트한다.  
     - 인코더(encoder)를 고정(freeze)하고 **헤드만 업데이트(only the head)** 한다.  
     - 인코더 대부분을 고정하고 **상위 층(top layers)** 과 **헤드(head)** 를 함께 업데이트한다.  

<img src="/assets/img/lecture/textmining/8/image_5.png" alt="image" width="800px">

---

# p15. 분류 작업(Classification task)

---

## p16. 분류: 이진 분류 vs 다중 클래스 분류 (binary vs. multi-class) 

- **분류(Classification)**: 미리 정의된 범주 집합 $C$에서 **이산적인 레이블**을 예측하는 것  

$$
x \;\longrightarrow\; f \;\longrightarrow\; y \in C
$$

- 이진 분류 (Binary classification, $\mid C \mid = 2$)  
  - 예: 스팸 탐지 (스팸 / 정상 메일)  
<img src="/assets/img/lecture/textmining/2/image_3.png" alt="image" width="480px">

- 다중 클래스 분류 (Multiclass classification, $\mid C \mid > 2$)  
  - 예: 이미지 분류 (고양이, 개, 말)  
<img src="/assets/img/lecture/textmining/2/image_4.png" alt="image" width="480px">

---

## p18. 분류: 분류기(classifier)

- 이제 우리는 (1) 사전학습된 모델을 사용하여 텍스트를 **벡터 표현(vector representation)** 으로 인코딩하고,  
  (2) 각 클래스에 대한 **확률(probability)** 을 계산할 수 있다.  

<img src="/assets/img/lecture/textmining/8/image_6.png" alt="image" width="800px">
