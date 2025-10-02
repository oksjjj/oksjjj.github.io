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

또 다른 주요 방법 범주는  
명시적 연관성 모델링(explicit association modeling)에 기반하여  
이상을 탐지하는 것이다.  

벡터 자기회귀(vector autoregression)와  
상태 공간 모델(state space models)이  
이 범주에 속한다.  

그래프(graph) 또한 명시적으로 연관성을 포착하는 데 사용되었다.  

즉, 서로 다른 시점(time points)을 정점(vertices)으로 하여  
시계열(time series)을 표현하고,  
랜덤 워크(random walk)를 통해 이상을 탐지하는 방식이다  
(Cheng et al., 2008; 2009).  

---

---

> **(블로그 추가 설명) 그래프 기반 이상 탐지 (Graph-based Anomaly Detection)**  
> 
> **1. 그래프 구성하기 (How to build the graph)**  
> - 시계열(time series)의 각 시점(time point)을 **정점(vertex)** 으로 둔다.  
> - 두 시점 사이의 **간선(edge)** 은 연관성(association)이나 유사성(similarity)으로 정의한다.  
>   - **시간 인접 기반(Local adjacency)**: $t$ 시점은 보통 $t-1$, $t+1$과 연결.  
>   - **K-최근접 이웃(K-NN) 기반**: 각 시점을 가장 유사한 K개의 시점과 연결.  
>   - **완전 연결(Fully connected)**: 모든 시점을 연결하되, 간선 가중치(weight)는 거리/유사도 함수로 조정.  
> - 간선의 가중치 $w_{ij}$ 는 예를 들어 다음과 같이 정의할 수 있다:  
>   $$
>   w_{ij} = \exp\!\left(-\|x_i - x_j\|^2\right)
>   $$  
>   여기서 $x_i$는 $i$번째 시점의 관측값(feature)이다.  
> 
> **2. 랜덤 워크(Random Walk)로 정상/이상 구분하기**  
> - 전이 확률(transition probability)은 간선 가중치에 비례한다:  
>   $$
>   P_{ij} = \frac{w_{ij}}{\sum_k w_{ik}}
>   $$  
>   즉, $i$ 시점에서 $j$ 시점으로 이동할 확률은 두 점의 유사성이 클수록 높다.  
> - **정상(normal) 시점**:  
>   - 여러 다른 시점과 강하게 연결되어 있음.  
>   - 랜덤 워크가 이 정점을 방문할 확률이 안정적이고, 주변 정점들로 분포가 균일하게 퍼진다.  
>   - 따라서 분포 $\pi_t$는 시간이 지나며 **안정적으로 수렴**한다.  
> - **이상(anomaly) 시점**:  
>   - 다른 시점들과의 연결이 약하거나 특정한 방향으로만 치우침.  
>   - 랜덤 워크가 이 정점을 거의 방문하지 않거나, 머물지 못하고 곧 이탈한다.  
>   - 결과적으로 $\pi_t$ 분포가 **불균형하게 왜곡**되어 정상 패턴과 확연히 구분된다.  
> 
> **3. 요약**  
> 그래프는 시계열 데이터의 "연결 구조"를 제공하고,  
> 랜덤 워크는 그 구조 위에서 정상과 이상을 구분하는 "탐색 절차" 역할을 한다.  
> 이 조합 덕분에 단순한 포인트 단위 오차 계산보다  
> 더 **구조적이고 전역적인 관점(global perspective)** 에서 이상 탐지가 가능하다.  

---

일반적으로 이러한 고전적 방법들은  
유의미한 표현(informative representations)을 학습하고  
세밀한 연관성(fine-grained associations)을 모델링하기 어렵다.  

최근에는 그래프 신경망(Graph Neural Network, GNN)이  
다변량 시계열(multivariate time series)에서  
여러 변수들 간의 동적 그래프(dynamic graph)를 학습하는 데 적용되었다  
(Zhao et al., 2020; Deng & Hooi, 2021).  

더 풍부한 표현력을 가지기는 하지만,  
이렇게 학습된 그래프는 여전히 **단일 시점(single time point)** 에 한정되어 있으며,  
이는 복잡한 시간적 패턴(complex temporal patterns)을 다루기에는 불충분하다.  

또한 부분 시퀀스(subsequence) 기반 방법들은  
부분 시퀀스들 간의 유사성(similarity)을 계산하여  
이상을 탐지한다 (Boniol & Palpanas, 2020).  

이러한 방법들은 더 넓은 시간적 맥락(wider temporal context)을 탐색할 수는 있지만,  
각 시점(time point)과 전체 시계열(whole series) 간의  
세밀한 시간적 연관성(fine-grained temporal association)은 포착하지 못한다.  

본 논문에서는 Transformer (Vaswani et al., 2017)를  
비지도 환경(unsupervised regime)에서의  
시계열 이상 탐지(time series anomaly detection)에 적용하였다.  

Transformer는 다양한 분야에서 큰 진전을 이루어왔다.  
예를 들어, 자연어 처리(natural language processing, Brown et al., 2020),  
컴퓨터 비전(machine vision, Liu et al., 2021),  
그리고 시계열(time series, Zhou et al., 2021) 등이 있다.  

이러한 성공은 전역 표현(global representation)과  
장기 관계(long-range relation)를 통합적으로 모델링하는  
Transformer의 강력한 능력에 기인한다.  

각 시점(time point)의 연관성 분포(association distribution)는  
시간적 맥락(temporal context)에 대해 더 유의미한 설명을 제공할 수 있다.  

이는 시계열(time series)의 주기(period)나 추세(trend)와 같은  
동적 패턴(dynamic patterns)을 드러낸다.  

우리는 위에서 설명한 연관성 분포(association distribution)를  
**시리즈-연관성(series-association)** 이라고 명명한다.  

이는 Transformer를 통해 원시 시계열(raw series)로부터  
발견될 수 있다.  

더 나아가 우리는, 이상(anomalies)은 드물고 정상 패턴(normal patterns)이 지배적이기 때문에  
이상(anomalies)이 전체 시계열(whole series)과 강한 연관성(strong associations)을  
형성하기 어렵다는 것을 관찰하였다.  

이상(anomalies)의 연관성은 인접한 시점(adjacent time points)에 집중되는데,  
이는 시계열의 연속성(continuity)으로 인해  
이웃한 시점들이 유사한 비정상 패턴(abnormal patterns)을  
포함할 가능성이 더 높기 때문이다.  

이러한 인접 집중(adjacent-concentration) 귀납적 편향(inductive bias)을  
**사전 연관성(prior-association)** 이라고 한다.  

대조적으로, 지배적인 정상 시점(normal time points)은  
인접한 영역에 국한되지 않고,  
전체 시계열(whole series)과의 유의미한 연관성(informative associations)을  
발견할 수 있다.  

이러한 관찰에 기반하여, 우리는 연관성 분포(association distribution)가 지니는  
정상(normal)과 이상(abnormal)의 고유한 구별 가능성(distinguishability)을  
활용하고자 한다.  

이로부터 각 시점(time point)에 대해 새로운 이상 기준(anomaly criterion)을 정의할 수 있는데,  
이는 각 시점의 **사전 연관성(prior-association)** 과  
**시리즈 연관성(series-association)** 사이의 거리를 정량화하여 얻어진다.  

우리는 이를 **연관성 불일치(Association Discrepancy)** 라고 명명한다.  

앞서 언급했듯이, 이상(anomalies)의 연관성은  
인접 집중(adjacent-concentrating)될 가능성이 더 크기 때문에,  
이상은 정상 시점(normal time points)보다  
더 작은 연관성 불일치(association discrepancy)를 보이게 된다.  

이전 방법들을 넘어, 우리는 Transformer를  
비지도 시계열 이상 탐지(unsupervised time series anomaly detection)에 도입하고,  
연관성 학습(association learning)을 위한 **Anomaly Transformer** 를 제안한다.  

연관성 불일치(Association Discrepancy)를 계산하기 위해,  
우리는 셀프 어텐션(self-attention) 메커니즘을  
**어노말리 어텐션(Anomaly-Attention)** 으로 새롭게 설계하였다.  

이 메커니즘은 **이중 분기(two-branch) 구조**를 가지며,  
각 시점(time point)의 **사전 연관성(prior-association)** 과  
**시리즈 연관성(series-association)** 을 각각 모델링한다.  

사전 연관성(prior-association)은 학습 가능한 가우시안 커널(learnable Gaussian kernel)을 사용하여  
각 시점(time point)의 인접 집중(adjacent-concentration) 귀납적 편향(inductive bias)을 표현한다.  

반면 시리즈 연관성(series-association)은  
원시 시계열(raw series)로부터 학습된  
셀프 어텐션 가중치(self-attention weights)에 해당한다.  

또한 두 분기(branch) 사이에는 **미니맥스 전략(minimax strategy)** 이 적용되며,  
이를 통해 연관성 불일치(Association Discrepancy)의  
정상(normal)과 이상(abnormal) 간 구별 가능성(distinguishability)을 증폭시킨다.  

나아가 이를 기반으로 새로운 **연관성 기반 기준(association-based criterion)** 을 도출할 수 있다.  

Anomaly Transformer는 세 가지 실제 응용(real applications)을 포함한  
여섯 가지 벤치마크(benchmarks)에서  
강력한 성능(strong results)을 달성하였다.  

본 논문의 기여(contributions)는 다음과 같이 요약된다:  

- 연관성 불일치(Association Discrepancy)에 대한 핵심 관찰에 기반하여,  
  우리는 **Anomaly-Attention 메커니즘**을 갖춘 **Anomaly Transformer** 를 제안한다.  
  이 모델은 사전 연관성(prior-association)과 시리즈 연관성(series-association)을  
  동시에 모델링하여 연관성 불일치(Association Discrepancy)를 구현할 수 있다.  

- 우리는 연관성 불일치(Association Discrepancy)의  
  정상(normal)과 이상(abnormal) 간 구별 가능성(distinguishability)을 강화하기 위해  
  **미니맥스 전략(minimax strategy)** 을 제안한다.  
  이를 바탕으로 새로운 **연관성 기반 탐지 기준(association-based detection criterion)** 을  
  추가적으로 도출한다.  

- Anomaly Transformer는 세 가지 실제 응용(real applications)에 대한  
  여섯 가지 벤치마크(benchmarks)에서  
  **최첨단(state-of-the-art) 이상 탐지 성능**을 달성하였다.  
  이는 광범위한 제거 실험(extensive ablations)과  
  통찰력 있는 사례 연구(insightful case studies)를 통해 입증되었다.  

---

## 2 관련 연구 (Related Work)  

### 2.1 비지도 시계열 이상 탐지 (Unsupervised Time Series Anomaly Detection)  

중요한 실제 문제(real-world problem)로서,  
비지도 시계열 이상 탐지(unsupervised time series anomaly detection)는  
광범위하게 연구되어 왔다.  

이상(anomaly) 판별 기준(determination criterion)에 따라 분류하면,  
해당 패러다임(paradigms)은 대체로  
**밀도 추정(density-estimation)**,  
**클러스터링 기반(clustering-based)**,  
**재구성 기반(reconstruction-based)**,  
**자기회귀 기반(autoregression-based)** 방법들을 포함한다.  

밀도 추정(density-estimation) 기반 방법에서는,  
대표적인 고전적 기법인 **지역 이상치 요인(Local Outlier Factor, LOF, Breunig et al., 2000)** 과  
**연결성 이상치 요인(Connectivity Outlier Factor, COF, Tang et al., 2002)** 이 있다.  

이들은 각각 **지역 밀도(local density)** 와  
**지역 연결성(local connectivity)** 을 계산하여  
이상치(outlier)를 판별한다.  

**DAGMM (Zong et al., 2018)** 과 **MPPCACD (Yairi et al., 2017)** 는  
가우시안 혼합 모델(Gaussian Mixture Model, GMM)을 결합하여  
표현(representations)의 밀도(density)를 추정한다.  

클러스터링 기반(clustering-based) 방법에서는,  
이상 점수(anomaly score)가 항상 **클러스터 중심(cluster center)까지의 거리(distance)** 로  
정식화된다.  

**SVDD (Tax & Duin, 2004)** 와 **Deep SVDD (Ruff et al., 2018)** 는  
정상 데이터에서 얻어진 표현들(representations)을  
하나의 밀집된 클러스터(compact cluster)로 모은다.  

**THOC (Shen et al., 2020)** 는  
계층적 클러스터링 메커니즘(hierarchical clustering mechanism)을 통해  
중간 층(intermediate layers)에서의 다중 스케일 시간적 특징(multi-scale temporal features)을 융합(fuse)한다.  

그리고 다층 거리(multi-layer distances)를 이용하여  
이상을 탐지한다.  

**ITAD (Shin et al., 2020)** 는  
분해된 텐서(decomposed tensors)에 대해  
클러스터링(clustering)을 수행한다.  

재구성 기반(reconstruction-based) 모델들은  
재구성 오차(reconstruction error)를 통해  
이상을 탐지하려고 시도한다.  

**Park et al. (2018)** 은 **LSTM-VAE 모델**을 제안했는데,  
이는 시간적 모델링(temporal modeling)을 위해 LSTM을 기반(backbone)으로 사용하고,  
재구성을 위해 변분 오토인코더(Variational AutoEncoder, VAE)를 활용한다.  

**OmniAnomaly (Su et al., 2019)** 는  
LSTM-VAE 모델을 정규화 흐름(normalizing flow)으로 확장하고,  
재구성 확률(reconstruction probabilities)을 이용하여  
이상을 탐지한다.  

**InterFusion (Li et al., 2021)** 은  
백본(backbone)을 **계층적 VAE(hierarchical VAE)** 로 새롭게 설계하여,  
여러 시계열(multiple series) 간의 **상호 의존성(inter-dependency)** 과  
내부 의존성(intra-dependency)을 동시에 모델링한다.  

**GANs (Goodfellow et al., 2014)** 역시  
재구성 기반 이상 탐지(reconstruction-based anomaly detection)에 활용되며  
(Schlegl et al., 2019; Li et al., 2019a; Zhou et al., 2019),  
적대적 정규화(adversarial regularization)로 작동한다.  

자기회귀 기반(autoregression-based) 모델들은  
예측 오차(prediction error)를 통해  
이상을 탐지한다.  

**VAR** 는 **ARIMA (Anderson & Kendall, 1976)** 를 확장한 모델로,  
시차 의존 공분산(lag-dependent covariance)에 기반하여  
미래를 예측한다.  

자기회귀 모델(autoregressive model)은  
LSTM으로 대체될 수도 있다 (Hundman et al., 2018; Tariq et al., 2019).  

본 논문의 특징은 새로운 **연관성 기반 기준(association-based criterion)** 에 있다.  

랜덤 워크(random walk)나 부분 시퀀스(subsequence) 기반 방법들  
(Cheng et al., 2008; Boniol & Palpanas, 2020)과 달리,  
우리의 기준(criterion)은 보다 유의미한 시점 간 연관성(time-point associations)을 학습하기 위해  
시간적 모델(temporal models)의 **공동 설계(co-design)** 를 통해 구현된다.  

---

---

> **(블로그 추가 설명) 시간적 모델의 공동 설계 (Co-design of Temporal Models)**  
> "공동 설계(co-design)"란 단일한 모델을 사용하는 대신,  
> 서로 다른 특성을 가진 두 개 이상의 모델을 **함께 설계하고 상호 보완적으로 학습**시키는 방법을 의미한다.  
> 
> Anomaly Transformer에서는  
> - **사전 연관성(prior-association)** 을 표현하는 모델 → 인접 시점(adjacent points)에 집중하도록 설계  
> - **시리즈 연관성(series-association)** 을 표현하는 모델 → 전체 시계열(global context)과의 연관성을 학습하도록 설계  
> 
> 이렇게 두 가지 관점을 **동시에 학습(co-design)** 함으로써,  
> 단일 모델로는 잡아내기 어려운 **세밀하고 풍부한 시점 간 연관성**을 포착할 수 있다.  
> 
> 즉, 공동 설계는 "지역적 패턴(local patterns)"과 "전역적 패턴(global patterns)"을  
> 함께 반영하도록 모델 구조를 설계하는 접근이다.  

---

### 2.2 시계열 분석을 위한 Transformer (Transformers for Time Series Analysis)  

최근 Transformer (Vaswani et al., 2017)는  
자연어 처리(natural language processing, Devlin et al., 2019; Brown et al., 2020),  
오디오 처리(audio processing, Huang et al., 2019),  
컴퓨터 비전(computer vision, Dosovitskiy et al., 2021; Liu et al., 2021) 등  
순차 데이터(sequential data) 처리에서 강력한 성능을 보여주었다.  

시계열 분석(time series analysis)에서는  
셀프 어텐션(self-attention) 메커니즘의 장점에 힘입어,  
Transformer가 신뢰할 수 있는 장기 시간 의존성(long-range temporal dependencies)을  
발견하는 데 사용되고 있다  
(Kitaev et al., 2020; Li et al., 2019b; Zhou et al., 2021; Wu et al., 2021).  

특히 시계열 이상 탐지(time series anomaly detection)에서는,  
**GTA (Chen et al., 2021)** 가 제안되었는데,  
이는 그래프 구조(graph structure)를 활용하여  
여러 IoT 센서 간의 관계를 학습하고,  

Transformer를 사용하여 시간적 모델링(temporal modeling)을 수행하며,  
재구성 기준(reconstruction criterion)을 통해  
이상을 탐지한다.  

기존 Transformer 활용 방식과 달리,  
**Anomaly Transformer** 는 연관성 불일치(association discrepancy)에 대한 핵심 관찰에 기반하여  
셀프 어텐션(self-attention) 메커니즘을  
**어노말리 어텐션(Anomaly-Attention)** 으로 새롭게 설계하였다.  
