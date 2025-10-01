---
layout: post
title: "[확률과 통계] 주피터 노트북 실습"
date: 2025-10-01 16:00:00 +0900
categories:
  - "대학원 수업"
  - "확률과 통계"
tags: []
---

> 출처: 확률과 통계 – 박성우 교수님, 고려대학교 (2025)

## 주성분 분석(Principal Component Analysis, PCA): 그림으로 보는 소개

이 노트북은 시각화를 곁들여 **주성분 분석(PCA)** 을 간단하고 예제 중심으로 소개한다.  
합성된 2차원 데이터셋에서 시작하여 고차원 예제로 확장하고, NumPy/SVD 기반 구현과 표준 라이브러리 구현을 비교한다.

## 학습 목표
- **PCA**를 단위 노름 제약 조건 하에서 투영된 분산을 최대로 하는 직교 선형 변환으로 이해한다.
- 특잇값 분해(SVD)와 공분산 행렬의 고유분해를 통해 PCA를 계산한다.
- 주성분 축, 설명된 분산, 저차원 투영, 그리고 복원(reconstruction)을 시각화한다.

## 수학적 정식화
중심화된 데이터 행렬 $X \in \mathbb{R}^{n \times d}$ 가 주어졌을 때,  
PCA는 다음을 만족하는 직교 단위 벡터 집합 $\lbrace w_k \rbrace_{k=1}^d$ 를 찾는다.

$$
\max_{\|w\|_2=1} \operatorname{Var}(Xw) = \frac{1}{n}\|Xw\|_2^2,
$$

여기서 연속된 벡터들은 서로 직교하도록 제약된다.  
동등하게, $\Sigma = \tfrac{1}{n}X^\top X$ 라 하면 $w_k$ 는 고유값이 큰 순서대로 정렬된 $\Sigma$의 고유벡터들이다.  

또한 $X = U S V^\top$ 가 thin SVD라면, 주성분 방향은 \(V\)의 열벡터들이고, 설명된 분산은 $S^2/n$ 이다.  

순위-$r$ PCA 근사는 다음과 같이 정의된다.

$$
X_r = Z_r W_r^\top, \quad 
Z_r = X W_r, \quad 
W_r = [w_1, \dots, w_r], \quad r \le d,
$$

이 근사는 모든 순위-$r$ 선형 복원 중에서 프로베니우스 오차(Frobenius error)를 최소화한다.


```python
# 필요한 라이브러리 불러오기
import numpy as np                  # 수치 계산을 위한 라이브러리
import matplotlib.pyplot as plt     # 시각화를 위한 라이브러리
from pathlib import Path            # 파일 및 디렉토리 경로 처리를 위한 라이브러리

# 난수 생성기를 초기화 (seed=7로 고정하여 재현 가능하게 함)
rng = np.random.default_rng(7)
```

## 1. 상관된 방향을 가진 합성 2차원 데이터

비등방성 공분산을 가진 2차원 가우시안 분포의 점 구름(cloud)을 생성하고, 산점도를 시각화한 뒤,  
SVD 기반 PCA로 계산된 주성분 축을 함께 표시한다.

```python
# -----------------------------
# 2D 합성 데이터 생성 및 PCA 시각화
# -----------------------------

# 표본 수
n = 300

# 평균 벡터 (원점 중심)
mu = np.array([0.0, 0.0])

# 선형 변환 행렬 (비등방성 공분산을 만들기 위함)
#   - 대각 성분의 크기 차이와 비대각 성분(1.2)이 축 사이의 상관을 유도함
A = np.array([[2.0, 1.2],
              [0.0, 0.4]])

# 표준정규(0, I)에서 n×2 표본을 생성한 뒤, A^T로 선형 변환하고 평균을 더해
# 비등방성 가우시안 점구름을 만듦
X2 = rng.standard_normal((n, 2)) @ A.T + mu

# 중심화: 각 축(열)에서 평균을 빼서 평균이 0이 되도록 만듦
# keepdims=True로 차원을 유지하여 브로드캐스팅이 안전하게 동작하도록 함
Xc = X2 - X2.mean(axis=0, keepdims=True)

# SVD 수행 (thin SVD)
# Xc = U S V^T
# - V의 열벡터가 주성분 방향(고유벡터)에 해당
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
W = Vt.T  # 주성분 방향 행렬 (열마다 하나의 주성분 벡터)

# 설명된 분산(explained variance):
#  S(특잇값)와의 관계에서 S^2 / n 이 각 주성분의 분산에 해당
explained = (S**2) / Xc.shape[0]

# 설명된 분산 비율(전체 분산 대비 비율)
ratio = explained / np.sum(explained)

# -----------------------------
# 시각화
# -----------------------------
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)

# 중심화된 점구름 산점도
ax.scatter(Xc[:, 0], Xc[:, 1], s=1)

# 원점에서 주성분 축을 그리기 위한 시작점
origin = np.zeros(2)

# 상위 2개 주성분 벡터를 그려서 축을 시각화
for k in range(2):
    # 주성분 방향 벡터 W[:, k]를 분산의 제곱근(표준편차)에 비례하도록 스케일
    # 보기 좋게 3배 확대하여 화살표 길이를 조정
    vec = W[:, k] * np.sqrt(explained[k]) * 3.0
    ax.plot([origin[0], vec[0]], [origin[1], vec[1]])

# 축비 동일하게 설정 (왜곡 없는 원형 스케일)
ax.set_aspect("equal", adjustable="box")

# 제목
ax.set_title("2D cloud with principal axes")

plt.show()
```

<img src="/assets/img/probstat/pr1/image_1.png" alt="image" width="600px">


## 2. SVD로 처음부터 구현하는 PCA

PCA를 위한 가벼운 유틸리티를 구현한다: 중심화(centering), SVD, \(r\)개 성분으로의 투영, 복원(reconstruction), 그리고 설명된 분산 비율(explained-variance ratio).

```python
# -----------------------------
# PCA를 SVD로 직접 구현한 함수
# -----------------------------

def pca_svd(X, r=None):
    # 1. 데이터 중심화 (각 열의 평균을 빼줌)
    Xc = X - X.mean(axis=0, keepdims=True)

    # 2. SVD 수행
    #    Xc = U S V^T
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    # 3. 사용할 주성분 개수 r 설정
    #    기본값은 전체 차원 수
    if r is None:
        r = Vt.shape[0]

    # 4. 주성분 방향 행렬 (상위 r개의 열벡터)
    W = Vt[:r].T

    # 5. 주성분 공간으로 투영한 데이터 (scores)
    Z = Xc @ W

    # 6. 설명된 분산 (고유값에 해당)
    explained = (S**2) / Xc.shape[0]

    # 7. 설명된 분산 비율 (전체 분산 대비 각 성분의 기여도)
    ratio = explained / np.sum(explained)

    # 투영 데이터 Z, 주성분 W, 중심화 데이터 Xc,
    # 설명된 분산 explained, 설명된 분산 비율 ratio 반환
    return Z, W, Xc, explained, ratio


# -----------------------------
# 복원 함수
# -----------------------------
def reconstruct(Z, W, X_mean):
    # 투영된 데이터 Z와 주성분 W, 평균을 이용해
    # 원래 공간으로 복원한 데이터 반환
    return Z @ W.T + X_mean
```

### 2.1 Scree plot (explained variance)

고유값의 크기와 누적 설명된 분산을 시각화한다.

```python
# -----------------------------
# Scree plot 및 누적 설명 분산 시각화
# -----------------------------

# PCA 수행 (상위 2개 성분)
Z, W, Xc2, expl2, ratio2 = pca_svd(X2, r=2)

# -----------------------------
# Scree plot (각 성분의 설명된 분산 크기)
# -----------------------------
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111)
ax.plot(np.arange(1, 3), expl2, marker="o")
ax.set_xlabel("component")          # 성분 번호
ax.set_ylabel("explained variance") # 설명된 분산
ax.set_title("Scree")
plt.show()

# -----------------------------
# 누적 설명 분산 비율
# -----------------------------
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111)
ax.plot(np.arange(1, 3), np.cumsum(ratio2), marker="o")
ax.set_xlabel("component")                 # 성분 번호
ax.set_ylabel("cumulative ratio")          # 누적 비율
ax.set_ylim(0, 1.05)                       # y축 범위 0 ~ 1.05
ax.set_title("Cumulative explained variance")
plt.show()
```

<img src="/assets/img/probstat/pr1/image_2.png" alt="image" width="480px">

<img src="/assets/img/probstat/pr1/image_3.png" alt="image" width="480px">


### 2.2 1차원 투영과 복원

첫 번째 성분에 데이터를 투영하고, 복원된 결과를 시각화한다.

```python
# -----------------------------
# 1차원 투영과 복원 시각화
# -----------------------------

# PCA 수행 (상위 1개 성분만 사용)
Z1, W1, Xc2b, expl2b, ratio2b = pca_svd(X2, r=1)

# 원래 데이터의 평균 계산
Xmean = X2.mean(axis=0, keepdims=True)

# 1차원 성분을 이용해 복원
Xr1 = reconstruct(Z1, W1, Xmean)

# -----------------------------
# 원본 데이터 vs rank-1 복원 결과 비교
# -----------------------------
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)

# 원본 데이터 산점도 (회색, 반투명)
ax.scatter(X2[:, 0], X2[:, 1], s=8, alpha=0.5)

# rank-1 복원 데이터 산점도 (파랑)
ax.scatter(Xr1[:, 0], Xr1[:, 1], s=8)

# 일부 점마다 원본과 복원을 연결하는 선 (25번째마다 하나씩)
for i in range(0, X2.shape[0], 25):
    ax.plot([X2[i, 0], Xr1[i, 0]],
            [X2[i, 1], Xr1[i, 1]])

# 축비 동일하게 설정
ax.set_aspect("equal", adjustable="box")

# 제목
ax.set_title("Original vs rank-1 reconstruction")

plt.show()
```

<img src="/assets/img/probstat/pr1/image_4.png" alt="image" width="600px">


## 3. 차원에 따른 복원 오차

비등방성 5차원 데이터셋에서, 보존된 순위(rank)가 증가함에 따라 프로베니우스(Frobenius) 복원 오차가 어떻게 감소하는지 살펴본다.

```python
# -----------------------------
# 5차원 데이터에서 차원(r)에 따른 복원 오차 확인
# -----------------------------

# 표본 수와 차원 수
n = 800
d = 5

# 직교 행렬 Q 생성 (QR 분해 이용)
Q, _ = np.linalg.qr(rng.standard_normal((d, d)))

# 고유값 스펙트럼 (3.0에서 0.2까지 선형적으로 감소)
vals = np.linspace(3.0, 0.2, d)

# 공분산 행렬 C = Q diag(vals) Q^T
C = Q @ np.diag(vals) @ Q.T

# 다변량 정규분포 표본 생성 (평균 0, 공분산 C)
X5 = rng.multivariate_normal(np.zeros(d), C, size=n)

# -----------------------------
# 순위 r에 따른 복원 오차 계산
# -----------------------------
errs = []

# 전체 분산 (평균을 뺀 뒤 Frobenius norm 제곱)
tot = np.linalg.norm(X5 - X5.mean(axis=0, keepdims=True))**2

# r=1부터 d까지 PCA 수행 후 복원 오차 측정
for r in range(1, d+1):
    Z, W, Xc, expl, ratio = pca_svd(X5, r=r)
    Xr = reconstruct(Z, W, X5.mean(axis=0, keepdims=True))
    e = np.linalg.norm(X5 - Xr)**2
    errs.append(e)

# -----------------------------
# 시각화: 상대적 복원 오차 (relative error)
# -----------------------------
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111)
ax.plot(range(1, d+1), np.array(errs)/tot, marker="o")
ax.set_xlabel("rank r")                       # 보존된 차원 수
ax.set_ylabel("relative reconstruction error") # 상대적 복원 오차
ax.set_title("Error decay")                   # 차원 증가에 따른 오차 감소
plt.show()
```

<img src="/assets/img/probstat/pr1/image_5.png" alt="image" width="480px">


## 4. 3D 시각화

$ \mathbb{R}^3 $에서 거의 1차원에 가까운 필라멘트 형태의 데이터를 합성하고, 점구름을 시각화한 뒤, 2차원 PCA 투영 결과와 비교한다.

```python
# -----------------------------
# 3차원 데이터 합성과 PCA 투영
# -----------------------------

# 표본 수
m = 700

# 매개변수 t ~ Uniform(-3, 3)
t = rng.uniform(-3, 3, size=m)

# 1차원 선형 구조 (필라멘트) 생성
line = np.stack([3*t, 0.5*t, -t], axis=1)

# 잡음 추가 (정규분포, 표준편차 0.7)
noise = rng.normal(scale=0.7, size=(m, 3))

# 최종 3차원 데이터셋
X3 = line + noise

# PCA 수행 (상위 2개 성분 유지)
Z, W, Xc, expl, ratio = pca_svd(X3, r=2)

# -----------------------------
# 3차원 데이터 점구름 시각화
# -----------------------------
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], s=6)
ax.set_title("3D cloud")
plt.show()

# -----------------------------
# 2차원 PCA 투영 결과 시각화
# -----------------------------
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111)

# 2차원 복원 데이터 (평균을 다시 더해줌)
Xproj = Z @ W.T + X3.mean(axis=0, keepdims=True)

# 산점도
ax.scatter(Xproj[:, 0], Xproj[:, 1], s=6)
ax.set_title("PCA 2D projection in first two coordinates")
plt.show()
```

<img src="/assets/img/probstat/pr1/image_6.png" alt="image" width="480px">

<img src="/assets/img/probstat/pr1/image_7.png" alt="image" width="480px">


## 5. 라이브러리 기반 PCA (옵션) — 고전 데이터셋

가능하다면 표준 구현을 사용하여 잘 알려진 데이터셋에 적용해 본다.


```python
# -----------------------------
# 라이브러리 기반 PCA (scikit-learn) 예제
# -----------------------------
try:
    from sklearn.decomposition import PCA
    from sklearn import datasets

    # -----------------------------
    # Iris 데이터셋 예제
    # -----------------------------
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # PCA: 2차원으로 축소
    pca = PCA(n_components=2, svd_solver="full")
    Xp = pca.fit_transform(X)

    # 시각화
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    ax.scatter(Xp[:, 0], Xp[:, 1], s=12, c=y)
    ax.set_title("Iris: PCA to 2D")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.show()

    # -----------------------------
    # Digits 데이터셋 예제
    # -----------------------------
    digits = datasets.load_digits()
    Xd = digits.data
    yd = digits.target

    # PCA: 2차원으로 축소 (랜덤화 SVD 사용)
    pca2 = PCA(n_components=2, svd_solver="randomized", random_state=0)
    Xpd = pca2.fit_transform(Xd)

    # 시각화
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    sc = ax.scatter(Xpd[:, 0], Xpd[:, 1], s=8, c=yd)
    ax.set_title("Digits: PCA to 2D")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.show()

# -----------------------------
# scikit-learn 미설치 환경 처리
# -----------------------------
except Exception as e:
    fig = plt.figure(figsize=(5, 2))
    ax = fig.add_subplot(111)
    ax.text(0.05, 0.5, "sklearn not available in this runtime", fontsize=12)
    ax.axis("off")
    plt.show()
```

<img src="/assets/img/probstat/pr1/image_8.png" alt="image" width="480px">

<img src="/assets/img/probstat/pr1/image_9.png" alt="image" width="480px">