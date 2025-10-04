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

---

> **(블로그 보충 설명) 셀프 어텐션 가중치 분포와 시계열의 연관성**  
> 셀프 어텐션(self-attention)은 입력 시퀀스 내의 각 시점(time point)이  
> 다른 모든 시점들과 얼마나 관련되어 있는지를 학습하는 메커니즘이다.  
> 
> 구체적으로, 한 시점의 **가중치 분포(weight distribution)** 는  
> 그 시점이 전체 시계열(whole series)의 어느 부분에 주의를 기울이고 있는지를 보여준다.  
> - 예를 들어, 주기적인 패턴(periodic pattern)이 있는 시계열에서는  
>   어텐션이 일정한 간격의 시점들에 반복적으로 집중하는 경향을 보인다.  
> - 반대로, 이상(anomaly)이 있는 구간에서는  
>   특정 시점의 어텐션 가중치가 불균형하게 쏠리거나 급격히 변한다.  
> 
> 이러한 이유로, **셀프 어텐션 가중치 분포**는  
> 단순히 값의 크기뿐만 아니라 시계열 내의 **구조적 관계(temporal structure)** 와  
> **패턴 간 상호작용(interactions among patterns)** 을 함께 반영한다.  
> 
> 따라서 Transformer를 활용하면  
> 포인트 단위 표현(pointwise representation)을 넘어  
> 시계열 전체의 전역적 연관성(global association)을 학습할 수 있다.  

---

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

## 1 서론 (Introduction)  

실세계(real-world) 시스템들은 항상 연속적인 방식으로 작동하며,  
산업 장비(industrial equipment), 우주 탐사선(space probe) 등과 같이  
다중 센서(multi-sensors)에 의해 모니터링되는  
여러 연속적인 측정값(successive measurements)을 생성한다.  

대규모 시스템 모니터링 데이터에서 오작동(malfunctions)을 발견하는 일은  
시계열(time series)에서 비정상 시점(abnormal time points)을 탐지하는 문제로 환원될 수 있으며,  
이는 보안을 보장하고(security) 재정적 손실(financial loss)을 방지하는 데 매우 중요하다.  

그러나 이상(anomalies)은 일반적으로 드물고,  
방대한 정상 지점(normal points)에 의해 가려져 있기 때문에  
데이터 라벨링(data labeling)은 어렵고 비용이 많이 든다.  

따라서 우리는 비지도 학습(unsupervised) 환경에서의  
시계열 이상 탐지(time series anomaly detection)에 초점을 맞춘다.  

비지도 시계열 이상 탐지(unsupervised time series anomaly detection)는  
실제 환경에서 매우 도전적인 문제이다.  

모델은 비지도 학습(unsupervised tasks)을 통해  
복잡한 시간적 동역학(complex temporal dynamics)으로부터  
유의미한 표현(informative representations)을 학습해야 한다.  

또한 풍부한 정상 시점(normal time points) 속에서  
드문 이상(rare anomalies)을 탐지할 수 있는  
구별 가능한 기준(distinguishable criterion)을 도출해야 한다.  

다양한 고전적 이상 탐지(classic anomaly detection) 방법들은  
여러 비지도 패러다임(unsupervised paradigms)을 제시해 왔다.  
예를 들어, **LOF (Local Outlier Factor, Breunig et al., 2000)** 에서 제안된  
밀도 추정(density-estimation) 기반 방법,  
**OC-SVM (One-Class SVM, Schölkopf et al., 2001)** 과  
**SVDD (Support Vector Data Description, Tax & Duin, 2004)** 에서 제시된  
클러스터링(clustering)-기반 방법 등이 있다.  

그러나 이러한 고전적 방법들은  
시간적 정보(temporal information)를 고려하지 않으며,  
보지 못한 실제 시나리오(unseen real scenarios)로 일반화하기 어렵다.  

신경망(neural networks)의 표현 학습 능력(representation learning capability)에 힘입어,  
최근의 딥러닝 기반 모델들(Su et al., 2019; Shen et al., 2020; Li et al., 2021)은  
우수한 성능(superior performance)을 달성하였다.  

이 중 주요한 접근 방식은  
잘 설계된 순환 신경망(recurrent networks)을 통해  
포인트 단위 표현(pointwise representations)을 학습하고,  
재구성(reconstruction) 또는 자기회귀(autoregressive) 과제를 통해  
자기지도(self-supervised) 방식으로 학습하는 것이다.  

이때 자연스럽고 실용적인 이상 기준(anomaly criterion)은  
포인트 단위의 재구성 오차(reconstruction error)나  
예측 오차(prediction error)이다.  

그러나 이상(anomalies)은 드물기 때문에,  
포인트 단위 표현(pointwise representation)은  
복잡한 시간적 패턴(complex temporal patterns)을 설명하기에는 충분하지 않으며,  
정상 시점(normal time points)에 의해 지배되어  
이상이 잘 구별되지 않는다.  

또한 재구성 오차나 예측 오차는  
포인트 단위(point by point)로 계산되기 때문에,  
시간적 맥락(temporal context)에 대한  
포괄적인 설명을 제공할 수 없다.  

또 다른 주요 방법 범주는  
명시적 연관성 모델링(explicit association modeling)에 기반하여  
이상을 탐지하는 것이다.  

벡터 자기회귀(vector autoregression)와  
상태 공간 모델(state space models)이 이 범주에 속한다.  

또한 그래프(graph) 역시 명시적인 연관성을 포착하는 데 사용되었다.  
서로 다른 시점(time points)을 정점(vertices)으로 하여 시계열(time series)을 표현하고,  
랜덤 워크(random walk)를 통해 이상을 탐지하는 방식이다 (Cheng et al., 2008; 2009).  

---

> **(블로그 추가 설명) 그래프와 랜덤 워크를 이용한 이상 탐지 방식**  
> 그래프 기반 이상 탐지(graph-based anomaly detection)는  
> 시계열(time series)의 **구조적 관계(structural relationship)** 를 명시적으로 표현하는 접근이다.  
> 
> 1. **그래프 구성 단계**  
>    - 각 시점(time point)을 그래프의 정점(vertex)으로 설정한다.  
>    - 시점 간의 유사도(similarity)나 상관관계(correlation)를 계산하여  
>      유사도가 높을수록 간선(edge)의 가중치(weight)를 크게 부여한다.  
>    - 예를 들어,  
>      $$
>      w_{ij} = \exp(-\|x_i - x_j\|^2)
>      $$  
>      와 같이 두 시점 $i, j$의 유사도에 기반해 간선 가중치를 정의할 수 있다.  
> 
> 2. **랜덤 워크(Random Walk) 수행**  
>    - 그래프 위에서 임의의 정점에서 시작해  
>      연결된 간선을 따라 확률적으로 이동하는 과정을 반복한다.  
>    - 정상 시점(normal points)은 다른 시점들과의 연결이 강하고 고르게 분포되어 있어,  
>      랜덤 워크가 안정적인 확률 분포(stationary distribution)에 수렴한다.  
>    - 반면 이상 시점(anomalies)은 연결이 약하거나 불균형하여  
>      방문 확률이 비정상적으로 낮거나 높게 나타난다.  
> 
> 3. **이상 탐지(Detection)**  
>    - 최종적으로 각 정점의 방문 확률 분포를 분석하여,  
>      정상 패턴과 확연히 다른 확률 특성을 보이는 정점을 이상으로 판단한다.  
> 
> 이러한 방식은 포인트 단위 오차(pointwise error)를 계산하는 방법보다  
> 더 **전역적(global)** 인 관점에서 시계열의 구조적 이상을 포착할 수 있다는 장점이 있다.  

---

일반적으로 이러한 고전적 방법들은  
유의미한 표현(informative representations)을 학습하고  
세밀한 연관성(fine-grained associations)을 모델링하기 어렵다.  

최근에는 그래프 신경망(Graph Neural Network, GNN)이  
다변량 시계열(multivariate time series)에서  
여러 변수 간의 동적 그래프(dynamic graph)를 학습하는 데 적용되었다  
(Zhao et al., 2020; Deng & Hooi, 2021).  

---

> **(블로그 추가 설명) 그래프 신경망(GNN)을 활용한 시계열 이상 탐지**  
> 그래프 신경망(Graph Neural Network, GNN)은  
> 그래프 구조에서 노드(정점) 간의 관계를 학습할 수 있는 신경망으로,  
> 시계열 데이터에서 **변수 간 상호의존성(inter-dependency)** 을 모델링하는 데 효과적이다.  
> 
> 1. **다변량 시계열에서의 그래프 구성**  
>    - 각 변수(variable)를 그래프의 정점(vertex)으로 설정한다.  
>      예를 들어, 센서 네트워크에서는 센서 하나가 하나의 정점이 된다.  
>    - 변수 간의 유사도(similarity), 상관계수(correlation), 또는 인과성(causality)을 기반으로  
>      간선(edge)을 정의한다.  
>    - 이렇게 구성된 그래프는 시간에 따라 변하므로 **동적 그래프(dynamic graph)** 로 표현된다.  
> 
> 2. **GNN의 학습 방식**  
>    - 각 정점은 자신과 인접한 노드들의 정보를 반복적으로 집계(aggregation)하여  
>      새로운 임베딩(embedding)을 생성한다.  
>      $$
>      h_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} w_{ij} \, h_j^{(l)}\right)
>      $$  
>      여기서 $h_i^{(l)}$는 $l$번째 층에서의 정점 $i$의 표현,  
>      $\mathcal{N}(i)$는 인접 노드 집합, $\sigma$는 비선형 활성화 함수이다.  
>    - 이 과정을 통해 모델은 변수들 간의 관계 구조를 단계적으로 학습한다.  
> 
> 3. **이상 탐지로의 적용**  
>    - 학습된 그래프 표현을 기반으로, 정상 상태에서는 강한 연결성을 보이지만  
>      이상이 발생하면 연결 구조가 급격히 약화되거나 변형된다.  
>    - 이러한 구조적 변화를 이상 지표(anomaly indicator)로 사용한다.  
> 
> 요약하자면, GNN은 시계열의 **시간 축**뿐 아니라  
> 변수 간의 **관계 축(relationship axis)** 까지 함께 고려함으로써,  
> 다변량 시계열에서의 복잡한 이상 패턴을 보다 정교하게 탐지할 수 있게 해준다.  

---

이러한 접근은 표현력이 더 풍부하긴 하지만,  
학습된 그래프는 여전히 **단일 시점(single time point)** 에 한정되어 있으며,  
복잡한 시간적 패턴(complex temporal patterns)을 다루기에는 불충분하다.  

또한 부분 시퀀스(subsequence) 기반 방법들은  
부분 시퀀스들 간의 유사성(similarity)을 계산하여  
이상을 탐지한다 (Boniol & Palpanas, 2020).  

---

> **(블로그 추가 설명) 부분 시퀀스(subsequence) 기반 이상 탐지 방식**  
> 부분 시퀀스(subsequence) 기반 방법은 시계열 전체를 한 번에 분석하는 대신,  
> 시계열을 일정 길이의 작은 구간(subsequence)으로 나누어  
> 각 구간 간의 유사성(similarity)을 비교함으로써 이상을 탐지하는 접근이다.  
> 
> 1. **부분 시퀀스 분할(Subsequence Segmentation)**  
>    - 길이가 $T$인 시계열을 슬라이딩 윈도우(sliding window) 기법을 이용해  
>      여러 개의 부분 시퀀스로 분할한다.  
>      예:  
>      $$
>      X = [x_1, x_2, \dots, x_T] \rightarrow 
>      \{[x_1,\dots,x_k], [x_2,\dots,x_{k+1}], \dots\}
>      $$  
>    - 이렇게 하면 시계열의 지역적(local) 패턴을 세밀하게 분석할 수 있다.  
> 
> 2. **유사성 계산(Similarity Computation)**  
>    - 각 부분 시퀀스 간의 거리를 계산하여 패턴의 유사성을 측정한다.  
>      자주 사용되는 거리 척도는 다음과 같다.  
>      - DTW (Dynamic Time Warping): 시간적 변형을 고려한 유사도 측정  
>      - Euclidean Distance: 단순한 시점별 차이 계산  
>    - 정상적인 부분 시퀀스들은 서로 높은 유사도를 보이지만,  
>      이상이 포함된 부분 시퀀스는 다른 구간들과의 유사도가 급격히 낮아진다.  
> 
> 3. **이상 탐지(Anomaly Detection)**  
>    - 각 부분 시퀀스의 평균 유사도 점수를 계산한 뒤,  
>      특정 임계값(threshold) 이하의 점수를 가진 시퀀스를 이상으로 판단한다.  
> 
> 이러한 방법은 시계열의 **지역적 이상(local anomaly)** 을 효과적으로 탐지할 수 있지만,  
> 시계열 전체의 전역적 구조(global temporal structure)를 반영하기 어렵다는 한계가 있다.  
> 특히, 시점 간 장기 의존성(long-term dependency)을 포착하기 힘들기 때문에  
> Transformer와 같은 전역적 모델의 필요성이 대두되었다.  

---

이러한 방법들은 더 넓은 시간적 맥락(wider temporal context)을 탐색할 수는 있지만,  
각 시점(time point)과 전체 시계열(whole series) 간의  
세밀한 시간적 연관성(fine-grained temporal association)은 포착하지 못한다.  

본 논문에서는 Transformer (Vaswani et al., 2017)를  
비지도 학습(unsupervised) 환경에서의 시계열 이상 탐지(time series anomaly detection)에 적용하였다.  

Transformer는 자연어 처리(natural language processing, Brown et al., 2020),  
머신 비전(machine vision, Liu et al., 2021),  
그리고 시계열 분석(time series, Zhou et al., 2021) 등 다양한 분야에서  
큰 성과를 거두었다.  

이러한 성공은 전역 표현(global representation)과  
장기 관계(long-range relation)를 통합적으로 모델링할 수 있는  
Transformer의 강력한 표현 능력에 기인한다.  

Transformer를 시계열에 적용한 결과,  
각 시점(time point)의 시간적 연관성(temporal association)은  
셀프 어텐션 맵(self-attention map)으로부터 얻어질 수 있음을 확인하였다.  
이는 시간 축(temporal dimension)을 따라  
각 시점이 전체 시계열의 다른 시점들과 맺는 연관성 가중치(association weights)의  
분포(distribution)로 표현된다.  

이러한 연관성 분포(association distribution)는  
시계열의 시간적 맥락(temporal context)에 대한  
더 풍부하고 유의미한 정보를 제공할 수 있으며,  
특히 시계열의 주기(period)나 추세(trend)와 같은  
동적 패턴(dynamic patterns)을 드러낸다.  

우리는 이러한 연관성 분포를 **시리즈 연관성(series-association)** 이라고 부르며,  
이는 Transformer를 통해 원시 시계열(raw series)로부터  
발견될 수 있다.  

더 나아가 우리는, 이상 지점(anomalies)들이 드물고  
정상 패턴(normal patterns)이 지배적이기 때문에,  
이상 지점이 전체 시계열(whole series)과 강한 연관성(strong associations)을  
형성하기 어렵다는 것을 관찰하였다.  

이상 지점의 연관성은 시계열의 연속성(continuity)으로 인해  
유사한 비정상 패턴(abnormal patterns)을 포함할 가능성이 높은  
인접한 시점(adjacent time points)에 집중되는 경향을 보인다.  

이러한 **인접 집중(adjacent-concentration)** 의 귀납적 편향(inductive bias)을  
우리는 **사전 연관성(prior-association)** 이라고 부른다.  

반면, 지배적인 정상 시점(normal time points)은  
인접한 구간에 한정되지 않고  
전체 시계열(whole series)과의 유의미한 연관성(informative associations)을  
형성할 수 있다.  

이러한 관찰에 기반하여, 우리는  
연관성 분포(association distribution)가 지니는  
정상과 이상 간의 고유한 구별 가능성(distinguishability)을 활용하고자 한다.  

이를 통해 각 시점(time point)에 대해  
새로운 이상 기준(anomaly criterion)을 정의할 수 있다.  
이 기준은 각 시점의 **사전 연관성(prior-association)** 과  
**시리즈 연관성(series-association)** 간의 거리를 정량화하여 산출되며,  
이를 **연관성 불일치(Association Discrepancy)** 라고 명명한다.  

앞서 언급했듯이,  
이상의 연관성은 인접한 시점에 집중되는 경향이 있기 때문에,  
이상 지점은 정상 시점에 비해  
더 작은 연관성 불일치(association discrepancy)를 나타내게 된다.  

이전의 방법들을 넘어, 우리는 Transformer를  
비지도 시계열 이상 탐지(unsupervised time series anomaly detection)에 도입하고,  
연관성 학습(association learning)을 위한 **Anomaly Transformer** 를 제안한다.  

연관성 불일치(Association Discrepancy)를 계산하기 위해,  
셀프 어텐션(self-attention) 메커니즘을 새롭게 설계하여  
**어노말리 어텐션(Anomaly-Attention)** 구조를 제안한다.  

이 메커니즘은 **이중 분기(two-branch)** 구조를 가지며,  
각 시점(time point)의 **사전 연관성(prior-association)** 과  
**시리즈 연관성(series-association)** 을 각각 모델링한다.  

사전 연관성은 학습 가능한 가우시안 커널(learnable Gaussian kernel)을 사용하여  
각 시점의 인접 집중(adjacent-concentration) 특성을 반영하고,  
시리즈 연관성은 원시 시계열(raw series)로부터 학습된  
셀프 어텐션 가중치(self-attention weights)에 대응한다.  

또한 두 분기(branch) 사이에는 **미니맥스 전략(minimax strategy)** 이 적용되며,  
이를 통해 연관성 불일치의 정상(normal)–이상(abnormal) 간  
구별 가능성(distinguishability)을 강화하고,  
새로운 연관성 기반 기준(association-based criterion)을 도출한다.  

Anomaly Transformer는 세 가지 실제 응용 분야(real applications)에 걸친  
여섯 가지 벤치마크(benchmarks)에서 우수한 성능(strong results)을 달성하였다.  

**본 논문의 주요 기여는 다음과 같다.**  

- 연관성 불일치(Association Discrepancy)에 대한 핵심 관찰에 기반하여,  
  사전 연관성과 시리즈 연관성을 동시에 모델링하는  
  어노말리 어텐션(Anomaly-Attention) 메커니즘을 갖춘  
  **Anomaly Transformer** 를 제안하였다.  

- 연관성 불일치의 정상–이상 간 구별 가능성을 강화하기 위해  
  **미니맥스 전략(minimax strategy)** 을 도입하고,  
  이를 통해 새로운 연관성 기반 이상 탐지 기준을 제시하였다.  

- 제안한 모델은 세 가지 실제 응용 분야의 여섯 가지 벤치마크에서  
  **최첨단(state-of-the-art)** 이상 탐지 성능을 달성하였으며,  
  이는 광범위한 제거 실험(ablation study)과  
  사례 분석(case study)을 통해 검증되었다.  

## 2 관련 연구 (Related Work)  

### 2.1 비지도 시계열 이상 탐지 (Unsupervised Time Series Anomaly Detection)  

중요한 실제 문제(real-world problem)로서,  
비지도 시계열 이상 탐지(unsupervised time series anomaly detection)는  
광범위하게 연구되어 왔다.  

이상(anomaly) 판별 기준(determination criterion)에 따라 분류하면,  
이러한 접근 방식들은 대체로  
**밀도 추정(density-estimation)**,  
**클러스터링 기반(clustering-based)**,  
**재구성 기반(reconstruction-based)**,  
그리고 **자기회귀 기반(autoregression-based)** 방법으로 구분된다.  

밀도 추정(density-estimation) 기반 방법에서는,  
대표적인 고전적 접근법인 **LOF (Local Outlier Factor, Breunig et al., 2000)** 과  
**COF (Connectivity Outlier Factor, Tang et al., 2002)** 가 있다.  

이들은 각각 **지역 밀도(local density)** 와  
**지역 연결성(local connectivity)** 을 계산하여  
이상치(outlier)를 판별한다.  

---

> **(블로그 추가 설명) 밀도 추정 기반 이상 탐지의 개념**  
> 밀도 추정(density estimation) 기반 방법은  
> **데이터가 얼마나 밀집되어 있는가(= 주변 데이터와 얼마나 가까운가)** 를 기준으로  
> 이상 여부를 판단하는 접근이다.  
> 
> 1. **기본 아이디어**  
>    - 정상(normal) 데이터는 주변에 유사한 점들이 많아 **밀도(density)** 가 높다.  
>    - 반면 이상치(outlier)는 주변에 유사한 점이 거의 없어 **밀도** 가 낮다.  
>    - 따라서 "밀도가 낮은 데이터"를 이상으로 간주한다.  
> 
> 2. **LOF (Local Outlier Factor)**  
>    - 각 데이터 포인트의 **지역 밀도(local density)** 를 계산하고,  
>      이웃들과 비교하여 얼마나 밀도가 떨어지는지를 평가한다.  
>    - LOF 값이 높을수록 주변보다 상대적으로 고립되어 있다는 뜻이며,  
>      이상치일 가능성이 크다.  
> 
> 3. **COF (Connectivity Outlier Factor)**  
>    - LOF와 유사하지만, 단순한 거리 대신  
>      **연결성(connectivity)** 을 기반으로 이상 여부를 판단한다.  
>    - 즉, 한 점이 다른 점들과 얼마나 잘 연결되어 있는지를 측정하여  
>      연결성이 약한 점을 이상으로 본다.  
> 
> 4. **한계점**  
>    - 이 접근은 데이터 간의 공간적 분포(spatial distribution)를 잘 반영하지만,  
>      **시간적 정보(temporal dependency)** 가 있는 시계열 데이터에는 그대로 적용하기 어렵다.  
>    - 그래서 시계열에서는 딥러닝 기반의 표현 학습(representation learning) 기법이 필요하게 되었다.  

---

또한 **DAGMM (Zong et al., 2018)** 과 **MPPCACD (Yairi et al., 2017)** 는  
가우시안 혼합 모델(Gaussian Mixture Model, GMM)을 결합하여  
표현 공간(representation space)에서의 밀도(density)를 추정한다.  

---

> **(블로그 추가 설명) DAGMM과 MPPCACD의 밀도 추정 방식**  
> **DAGMM (Deep Autoencoding Gaussian Mixture Model)** 과  
> **MPPCACD (Multi-Process Principal Component Analysis Change Detection)** 는  
> 고차원 데이터의 **은닉 표현(hidden representation)** 공간에서  
> 데이터 분포의 밀도를 추정함으로써 이상을 탐지하는 대표적인 딥러닝 기반 밀도 추정 기법이다.  
> 
> 1. **DAGMM의 핵심 아이디어**  
>    - 입력 데이터를 오토인코더(autoencoder)를 통해 저차원 잠재 공간(latent space)으로 압축한다.  
>    - 이 잠재 표현(latent representation)을 이용해  
>      가우시안 혼합 모델(Gaussian Mixture Model, GMM)을 학습한다.  
>    - 정상 데이터는 GMM의 고밀도 영역(high-density region)에 위치하지만,  
>      이상치는 저밀도 영역(low-density region)에 위치하게 된다.  
>    - 따라서 샘플의 **밀도 확률(density probability)** 이 낮을수록  
>      이상으로 판단할 수 있다.  
> 
> 2. **MPPCACD의 접근 방식**  
>    - 시계열 데이터를 여러 프로세스(subprocess)로 분리한 뒤,  
>      각 프로세스에 대해 주성분 분석(PCA)을 적용한다.  
>    - 이렇게 얻은 다중 프로세스 표현을 기반으로  
>      데이터 분포의 변화를 감지(change detection)하고,  
>      밀도 변동이 큰 구간을 이상으로 탐지한다.  
> 
> 3. **공통적인 특징**  
>    - 두 방법 모두 단순한 거리 기반 이상 탐지보다 더 확률적이고 정교한 접근을 취한다.  
>    - 특히 DAGMM은 딥러닝을 통해 특징 공간을 자동 학습하므로,  
>      기존 GMM보다 복잡한 데이터 분포를 잘 포착할 수 있다.  
> 
> 4. **한계점**  
>    - 학습 데이터에 이상치가 포함될 경우 GMM의 분포가 왜곡될 수 있으며,  
>      시계열의 시간적 순서를 직접적으로 반영하지 못한다는 점에서  
>      Transformer 기반 접근법과 같은 **시계열 구조적 모델링**이 필요하다.  

---

클러스터링 기반(clustering-based) 방법에서는  
이상 점수(anomaly score)를 일반적으로 **클러스터 중심(cluster center)** 까지의  
거리(distance)로 정의한다.  

**SVDD (Tax & Duin, 2004)** 와 **Deep SVDD (Ruff et al., 2018)** 는  
정상 데이터(normal data)에서 얻어진 표현들(representations)을  
하나의 밀집된 클러스터(compact cluster)로 모은다.  

---

> **(블로그 추가 설명) SVDD와 Deep SVDD의 이상 탐지 원리**  
> **SVDD (Support Vector Data Description)** 와 **Deep SVDD** 는  
> 정상(normal) 데이터가 어떤 “공간적 경계(boundary)” 안에 존재한다는 가정 하에  
> 이를 기반으로 이상을 판별하는 대표적인 **클러스터링 기반 이상 탐지 기법**이다.  
> 
> 1. **SVDD의 기본 아이디어**  
>    - 정상 데이터 포인트들이 포함되는 **최소 구(minimum hypersphere)** 를 찾는다.  
>    - 구의 중심을 $c$, 반지름을 $R$이라 하면,  
>      다음 최적화 문제를 푼다:  
>      $$
>      \min_{R, c} \; R^2 + C \sum_i \xi_i \quad \text{s.t.} \quad \|x_i - c\|^2 \le R^2 + \xi_i, \; \xi_i \ge 0
>      $$  
>    - 즉, 가능한 한 작은 구로 대부분의 정상 데이터를 감싸면서,  
>      구 밖에 위치한 포인트는 이상치로 간주한다.  
> 
> 2. **Deep SVDD의 확장**  
>    - 입력 데이터를 신경망을 통해 잠재 공간(latent space)으로 변환한 뒤,  
>      그 공간에서 SVDD의 원리를 적용한다.  
>    - 이를 통해 복잡한 비선형 경계를 학습할 수 있어,  
>      단순한 선형 구(sphere)보다 훨씬 강력한 표현력을 가진다.  
>    - 학습된 신경망은 정상 데이터의 표현을  
>      하나의 **밀집된 중심 영역(compact cluster)** 으로 모으는 방향으로 파라미터를 조정한다.  
> 
> 3. **핵심 아이디어 요약**  
>    - 정상 데이터는 중심 근처의 밀집된 영역에 분포하고,  
>      이상 데이터는 이 경계 밖(outside the boundary)에 존재한다.  
>    - 따라서 새로운 샘플이 학습된 구의 경계 밖에 위치하면  
>      이상치(anomaly)로 판별된다.  
> 
> 4. **한계점**  
>    - SVDD 계열 방법은 데이터 간 시간적 관계를 고려하지 않기 때문에  
>      시계열 데이터의 동적 특성을 모델링하기에는 한계가 있다.  
>      이런 점에서 Transformer 기반 모델은  
>      “정상 패턴의 공간적 경계”뿐 아니라  
>      “시간적 연관성(temporal association)”까지 함께 학습할 수 있다는 강점을 가진다.  

---

**THOC (Shen et al., 2020)** 는  
계층적 클러스터링 메커니즘(hierarchical clustering mechanism)을 이용해  
중간 계층(intermediate layers)에서의 다중 스케일 시간적 특징(multi-scale temporal features)을  
통합(fuse)하고,  
다층 간 거리(multi-layer distances)를 기반으로 이상을 탐지한다.  

---

> **(블로그 추가 설명) THOC의 계층적 이상 탐지 구조**  
> **THOC (Temporal Hierarchical One-Class Network)** 은  
> 시계열 데이터에서 서로 다른 시간적 범위의 패턴을 포착하기 위해  
> **계층적 클러스터링(hierarchical clustering)** 개념을 신경망 구조에 통합한 모델이다.  
> 
> 1. **핵심 아이디어**  
>    - 시계열 데이터는 단일 시간 스케일로만 보면 놓치기 쉬운  
>      다양한 수준(level)의 패턴 — 예를 들어,  
>      짧은 주기(short-term) 진동부터 긴 주기(long-term) 추세까지 — 를 포함한다.  
>    - THOC는 여러 계층(intermediate layers)에서 추출된 특징들을  
>      통합적으로 분석하여 이러한 **다중 스케일(multi-scale)** 패턴을 학습한다.  
> 
> 2. **모델 구조**  
>    - 하위 계층(lower layer)은 빠르게 변하는 단기 패턴(short-term dynamics)을,  
>      상위 계층(higher layer)은 느린 장기 패턴(long-term dependencies)을 포착한다.  
>    - 각 계층의 표현(feature representations)을 **클러스터링(clustering)** 하여  
>      정상 데이터가 형성하는 군집(cluster) 구조를 파악한다.  
> 
> 3. **이상 탐지 방식**  
>    - 입력 시계열을 각 계층의 군집 구조에 투영한 뒤,  
>      “해당 계층의 중심(cluster center)”으로부터의 거리(distance)를 계산한다.  
>    - 모든 계층의 거리 정보를 통합하여  
>      **다층 간 거리(multi-layer distances)** 로 이상 점수를 산출한다.  
>    - 정상 데이터는 여러 계층에서 일관된 군집 중심에 가깝게 위치하지만,  
>      이상 데이터는 특정 계층에서 중심으로부터 멀어지거나  
>      계층 간 일관성이 깨지는 형태를 보인다.  
> 
> 4. **특징과 의의**  
>    - THOC는 기존 단일 스케일 모델이 포착하지 못한  
>      복합적인 시간적 구조(temporal hierarchy)를 반영한다.  
>    - 즉, **시간적 다층 표현(time-scale hierarchy)** 을 활용하여  
>      더 안정적이고 세밀한 이상 탐지가 가능하다.  
> 
> 5. **한계점**  
>    - 그러나 여전히 포인트 간의 명시적 연관성(association)을 직접 모델링하지 않으며,  
>      Transformer 기반의 **전역적 어텐션(attention)** 메커니즘처럼  
>      시점 간 관계를 정량적으로 표현하지는 못한다.  

---

**ITAD (Shin et al., 2020)** 는  
분해된 텐서(decomposed tensors)에 대해 클러스터링(clustering)을 수행한다.  

---

> **(블로그 추가 설명) ITAD의 텐서 분해 기반 이상 탐지**  
> **ITAD (Interpretable Tensor-based Anomaly Detection)** 는  
> 다차원 시계열(multivariate time series)을  
> **텐서(tensor)** 형태로 표현하고,  
> 그 내부 구조를 분해(decomposition)하여 이상을 탐지하는 방법이다.  
> 
> 1. **텐서 표현 (Tensor Representation)**  
>    - 시계열 데이터를 3차원 텐서로 변환한다.  
>      예를 들어,  
>      - 첫 번째 축: 시간(time)  
>      - 두 번째 축: 변수(feature or sensor)  
>      - 세 번째 축: 관측 구간(window or segment)  
>    - 이렇게 하면 데이터의 **시간적(time)**, **공간적(spatial)**, **변수 간 관계(inter-variable)** 를  
>      하나의 구조 안에 함께 표현할 수 있다.  
> 
> 2. **텐서 분해 (Tensor Decomposition)**  
>    - 텐서 내부의 패턴을 저차원 성분(low-rank components)으로 분리하여  
>      주요 구조(main structure)와 노이즈(noise)를 구분한다.  
>    - 대표적인 방법은 **CP 분해(Canonical Polyadic Decomposition)** 또는  
>      **Tucker 분해(Tucker Decomposition)** 로,  
>      데이터의 핵심 패턴(core pattern)을 추출할 수 있다.  
> 
> 3. **클러스터링 및 이상 탐지 (Clustering and Detection)**  
>    - 분해된 저차원 표현(latent representation)을 클러스터링하여  
>      정상 데이터의 구조적 패턴을 학습한다.  
>    - 새로운 데이터가 기존 클러스터 구조와 크게 다를 경우,  
>      그 점을 이상(anomaly)으로 판단한다.  
> 
> 4. **의의와 장점**  
>    - ITAD는 단순히 개별 시점(point)이나 쌍(pair) 수준이 아니라,  
>      **시계열 전체의 다차원적 구조(multidimensional structure)** 를 반영할 수 있다.  
>    - 특히 “어떤 축(시간, 변수, 구간)”에서 이상이 발생했는지  
>      해석 가능한(interpretable) 형태로 파악할 수 있다는 점에서 의미가 크다.  
> 
> 5. **한계점**  
>    - 텐서 분해는 계산 비용이 크고,  
>      복잡한 시계열에서의 비선형 관계를 충분히 포착하기 어렵다.  
>    - 따라서 최근 연구에서는 Transformer 기반 모델처럼  
>      **비선형적이고 전역적인 연관성(global association)** 을 학습할 수 있는 접근이 주목받고 있다.  

---