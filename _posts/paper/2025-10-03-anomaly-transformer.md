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