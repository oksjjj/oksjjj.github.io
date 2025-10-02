---
layout: post
title: "[논문] Anomaly Transformer"
date: 2025-10-03 05:00:00 +0900
categories:
  - "논문"
tags: []
---

> **논문 출처**  
> Xu, J., Wu, H., Wang, J., & Long, M.  
> *Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy*.  
> International Conference on Learning Representations (ICLR 2022).  
> <a href="https://arxiv.org/abs/2110.02642" target="_blank">🔗 원문 링크 (arXiv:2110.02642)</a>

# ANOMALY TRANSFORMER: TIME SERIES ANOMALY DETECTION WITH ASSOCIATION DISCREPANCY  

**저자**  
- Jiehui Xu (Tsinghua University, BNRist, School of Software) - xjh20@mails.tsinghua.edu.cn  
- Haixu Wu (Tsinghua University, BNRist, School of Software) - whx20@mails.tsinghua.edu.cn  
- Jianmin Wang (Tsinghua University, BNRist, School of Software) - jimwang@tsinghua.edu.cn  
- Mingsheng Long (Tsinghua University, BNRist, School of Software) - mingsheng@tsinghua.edu.cn  

---

**주석**  
  
∗ 공동 기여(Equal contribution).

---

## 초록 (Abstract)  

시계열(time series)에서의 이상 지점(anomaly points)의 비지도 탐지는  
도전적인 문제이다.  

이는 모델이 **구별 가능한 기준(distinguishable criterion)** 을  
도출해내는 것을 요구한다.  

이전 방법들은 주로 **포인트 단위 표현(pointwise representation)**  
또는 **쌍별 연관성(pairwise association)** 을 학습함으로써  
이 문제를 다루어 왔다.  

그러나 어느 쪽도 복잡한 동역학(intricate dynamics)을  
추론하기에는 충분하지 않다.  

최근 Transformer는 **포인트 단위 표현(pointwise representation)** 과  
**쌍별 연관성(pairwise association)** 을 통합적으로 모델링하는 데에  
뛰어난 성능을 보여주었다.  

그리고 우리는 각 시점(time point)의 **셀프 어텐션 가중치 분포(self-attention weight distribution)** 가  
전체 시계열과의 풍부한 연관성(rich association)을 담을 수 있음을 발견했다.  

우리의 핵심 관찰은, 이상(anomalies)이 드물다는 이유로  
비정상 지점(abnormal points)에서 전체 시계열로 향하는  
비자명한 연관성(nontrivial associations)을 구축하기가  
극도로 어렵다는 것이다.  

따라서 이상 지점들의 연관성은 주로  
그들의 인접한 시점(adjacent time point)에 집중될 것이다.  

이러한 **인접 집중 편향(adjacent-concentration bias)** 은  
정상(normal) 지점과 비정상(abnormal) 지점을 본질적으로 구별할 수 있는  
**연관성 기반 기준(association-based criterion)** 을 내포한다.  

우리는 이를 **연관성 불일치(Association Discrepancy)** 를 통해 강조한다.  

기술적으로, 우리는 **연관성 불일치(association discrepancy)** 를 계산하기 위해  
새로운 **Anomaly-Attention 메커니즘**을 갖춘  
**Anomaly Transformer** 를 제안한다.  

정상(normal)과 비정상(abnormal) 사이에서  
연관성 불일치(association discrepancy)의 구별 가능성(distinguishability)을  
증폭하기 위해 **미니맥스 전략(minimax strategy)** 이 고안되었다.  

Anomaly Transformer는 세 가지 응용 분야, 즉 **서비스 모니터링(service monitoring)**,  
**우주 및 지구 탐사(space & earth exploration)**, **수자원 관리(water treatment)** 에서의  
여섯 가지 비지도 시계열 이상 탐지 벤치마크에서  
**최첨단(state-of-the-art) 성능**을 달성하였다.  

---

## 1 서론 (Introduction)  

실세계(real-world) 시스템들은 항상 연속적으로 동작하며,  
산업 장비(industrial equipment), 우주 탐사선(space probe) 등과 같이  
다중 센서(multi-sensors)에 의해 모니터링되는  
여러 연속적인 측정값(successive measurements)을 생성할 수 있다.  

대규모 시스템 모니터링 데이터에서 오작동(malfunctions)을 발견하는 것은  
시계열(time series)에서 비정상 시점(abnormal time points)을 탐지하는 문제로  
환원될 수 있으며, 이는 보안(security)을 보장하고  
재정적 손실(financial loss)을 피하는 데 매우 중요한 의미를 가진다.  

그러나 이상(anomalies)은 보통 드물고 방대한 정상 지점(normal points)에 의해 가려지기 때문에,  
데이터 라벨링(data labeling)은 어렵고 비용이 많이 든다(expensive).  

따라서 우리는 비지도 설정(unsupervised setting)에서의  
시계열 이상 탐지(time series anomaly detection)에 집중한다.  

비지도 시계열 이상 탐지(unsupervised time series anomaly detection)는  
실제(practice)에서 극도로 도전적인 과제이다.  

모델은 비지도 학습(unsupervised tasks)을 통해  
복잡한 시간적 동역학(complex temporal dynamics)으로부터  
유의미한 표현(informative representations)을 학습해야 한다.  

또한 모델은 풍부한 정상 시점(normal time points) 속에서  
드문 이상(rare anomalies)을 탐지할 수 있는  
구별 가능한 기준(distinguishable criterion)도 도출해야 한다.  

다양한 고전적인 이상 탐지(classic anomaly detection) 방법들은  
많은 비지도 학습(unsupervised) 패러다임을 제공해왔다.  

예를 들어, **지역 이상치 요인(Local Outlier Factor, LOF, Breunig et al., 2000)** 에서 제안된  
밀도 추정(density-estimation) 기반 방법,  

**원클래스 SVM(One-Class SVM, OC-SVM, Schölkopf et al., 2001)** 과  
**서포트 벡터 데이터 기술(Support Vector Data Description, SVDD, Tax & Duin, 2004)** 에서 제시된  
클러스터링(clustering) 기반 방법 등이 있다.  

이러한 고전적 방법들은 시간적 정보(temporal information)를 고려하지 않으며,  
보지 못한 실제 상황(unseen real scenarios)으로 일반화하기 어렵다.  

신경망(neural networks)의 표현 학습 능력(representation learning capability)에 힘입어,  
최근의 딥러닝 기반 모델들(Su et al., 2019; Shen et al., 2020; Li et al., 2021)은  
우수한 성능(superior performance)을 달성하였다.  

주요한 방법 범주 중 하나는, 잘 설계된 순환 신경망(recurrent networks)을 통해  
**포인트 단위 표현(pointwise representations)** 을 학습하는 데 집중한다.  

그리고 이들은 재구성(reconstruction) 또는 자기회귀(autoregressive) 과제를 통해  
자기지도(self-supervised) 방식으로 학습된다.  

여기에서 자연스럽고 실용적인 이상 기준(anomaly criterion)은  
포인트 단위(pointwise) **재구성 오차(reconstruction error)** 또는  
**예측 오차(prediction error)** 이다.  

그러나 이상(anomalies)이 드물기 때문에,  
포인트 단위 표현(pointwise representation)은 복잡한 시간적 패턴(complex temporal patterns)에 대해  
정보량이 부족하며(less informative),  

정상 시점(normal time points)에 의해 지배되어  
이상이 덜 구별 가능(less distinguishable)하게 될 수 있다.  

또한 재구성 오차(reconstruction error)나 예측 오차(prediction error)는  
포인트 단위(point by point)로 계산되기 때문에,  
시간적 맥락(temporal context)에 대한 포괄적인 설명을 제공할 수 없다.  
