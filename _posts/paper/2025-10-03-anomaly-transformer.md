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

시계열(time series)에서 이상 지점(anomaly points)을  
비지도 학습 방식으로 탐지하는 것은 도전적인 문제이며,  
이는 모델이 구별 가능한 기준(distinguishable criterion)을  
도출할 수 있도록 요구한다.  

이전 방법들은 주로 포인트 단위 표현(pointwise representation)이나  
쌍별 연관성(pairwise association)을 학습하는 방식으로 이 문제에 접근해 왔다.  

그러나 어느 쪽도 복잡한 동적 패턴(intricate dynamics)을 설명하기에는 충분하지 않다.  

---

> **(블로그 추가 설명) 포인트 단위 표현과 쌍별 연관성**  
> - **포인트 단위 표현 (Pointwise Representation)**  
>   각 시점(time point)을 독립적으로 표현하는 방식이다.  
>   예를 들어, 한 시점의 센서 값이나 그 주변 몇 개의 값만을 사용하여 특징 벡터를 만든다.  
>   → 문제: 각 시점을 따로 보므로, 전체적인 시간적 맥락을 반영하기 어렵다.  
> 
> - **쌍별 연관성 (Pairwise Association)**  
>   두 시점 사이의 관계를 모델링하는 방식이다.  
>   예를 들어, $t$ 시점과 $t+k$ 시점 사이의 상관관계(correlation)나 유사도(similarity)를 계산한다.  
>   → 문제: 쌍 단위 관계만 보므로, 더 복잡한 다중 시점 간의 상호작용(dynamics)을 포착하기 어렵다.  
> 
> **정리**  
> 포인트 단위 표현은 너무 "국소적(local)"이고,  
> 쌍별 연관성은 "이원적(pairwise)" 관계만 본다는 한계가 있다.  
> 따라서 복잡한 시간적 패턴(intricate temporal dynamics)을 이해하기 위해서는  
> 이보다 더 정교한, 다중 시점(multi-point) 수준의 연관성 학습이 필요하다.  

---

최근 Transformer는 포인트 단위 표현(pointwise representation)과  
쌍별 연관성(pairwise association)을 통합적으로 모델링하는 데에  
뛰어난 성능을 보여주었다.  

그리고 우리는 각 시점(time point)의 **셀프 어텐션 가중치 분포(self-attention weight distribution)** 가  
전체 시계열(whole series)과의 풍부한 연관성(rich association)을  
내포할 수 있음을 발견하였다.  

우리의 핵심 관찰은, 이상 지점(anomalies)들이 드물기 때문에  
이상 지점에서 전체 시계열(whole series)로의  
**자명하지 않은 연관성(nontrivial associations)** 을 형성하는 것이 극도로 어렵다는 것이다.  

따라서 이상 지점들의 연관성은 주로 인접한 시점(adjacent time points)에  
집중하게 된다.  

이웃한 시점에 연관성이 집중되는 경향(adjacent-concentration bias)은  
정상(normal) 지점과 이상(abnormal) 지점을 본질적으로 구별할 수 있는  
**연관성 기반 기준(association-based criterion)** 을 의미한다.  

우리는 이를 **연관성 불일치(Association Discrepancy)** 를 통해 강조한다.  

기술적으로, 우리는 연관성 불일치(association discrepancy)를 계산하기 위한  
새로운 Anomaly-Attention 메커니즘을 갖춘 Anomaly Transformer를 제안한다.  

연관성 불일치(association discrepancy)의  
정상(normal)과 이상(abnormal) 간 구별 가능성을 강화하기 위해  
미니맥스 전략(minimax strategy)이 고안되었다.  

Anomaly Transformer는 세 가지 응용 분야, 즉 서비스 모니터링(service monitoring),  
우주 및 지구 탐사(space & earth exploration), 수자원 관리(water treatment)에서의  
여섯 가지 비지도 시계열 이상 탐지 벤치마크에서  
최첨단(state-of-the-art) 성능을 달성하였다.  

---