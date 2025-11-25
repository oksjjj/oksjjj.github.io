---
layout: post
title: "[논문] Active Retrieval Augmented Generation"
date: 2025-11-23 23:00:00 +0900
categories:
  - "논문"
tags: []
---
> 논문 출처  
> Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu,  
> Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, Graham Neubig.  
> **능동 검색 기반 생성(Active Retrieval Augmented Generation).**  
> 카네기멜런대학교 언어기술연구소¹, Sea AI Lab², FAIR Meta³.  
> <a href="https://arxiv.org/abs/2305.06983" target="_blank">🔗 원문 링크 (arXiv: 2305.06983)</a>

저자  
- Zhengbao Jiang¹<sup>*</sup>  
- Frank F. Xu¹<sup>*</sup>  
- Luyu Gao¹<sup>*</sup>  
- Zhiqing Sun¹<sup>*</sup>  
- Qian Liu²  
- Jane Dwivedi-Yu³  
- Yiming Yang¹  
- Jamie Callan¹  
- Graham Neubig¹  

(¹Language Technologies Institute, Carnegie Mellon University  
 ²Sea AI Lab  
 ³FAIR, Meta)

> <sup>*</sup>주요 기여자(Lead contributors)이다.  

---

## 초록 (Abstract)

대규모 언어 모델(LMs)이 언어를 이해하고 생성하는 놀라운 능력에도 불구하고,  
LMs는 환각을 일으키고 사실적으로 부정확한 출력을 만들어내는 경향이 있다.  

외부 지식 자원으로부터 정보를 검색하여 LMs를 보완하는 것은  
하나의 유망한 해결책이다.  

대부분의 기존 검색 기반 증강 LMs는  
입력에 기반하여 단 한 번만 정보를 검색하는  
retrieve-and-generate 방식을 사용한다.  

그러나 이것은 긴 텍스트를 생성하는  
보다 일반적인 시나리오에서는 한계가 있다.  

그러한 경우에는 생성 과정 전체에 걸쳐  
지속적으로 정보를 수집하는 것이 필수적이다.  

이 연구에서 우리는  
능동 검색 기반 생성(active retrieval augmented generation)에 대한  
일반화된 관점을 제시한다.  

이 방법들은 생성 과정 전반에서 언제, 무엇을 검색해야 하는지를 능동적으로 결정한다.  

우리는 FLARE(Forward-Looking Active REtrieval augmented generation)를 제안한다.  

이 방식은 다음 문장에 대한 예측을 반복적으로 사용하여 향후 내용을 미리 예상하고,  
그 예측된 문장을 관련 문서들을 검색하기 위한 쿼리로 활용하여  
만약 낮은 신뢰도의 토큰이 포함될 경우  
해당 문장을 재생성하는 일반적(generic) 방법이다.  

우리는 FLARE를 4개의 장문 지식 집약적 생성 과제/데이터셋 전반에서  
기존 베이스라인들과 함께 종합적으로 테스트하였다.  

FLARE는 모든 과제에서 뛰어나거나 경쟁력 있는 성능을 달성하였으며,  
이는 우리의 방법<sup>1</sup>의 효과성을 입증한다.

> <sup>1</sup>코드와 데이터셋은 https://github.com/jzbjyb/FLARE 에서 제공된다.

---

## 1 서론 (Introduction)

생성 언어 모델(LMs)은  
(Brown et al., 2020; Ouyang et al., 2022; OpenAI, 2023;  
Chowdhery et al., 2022; Zhang et al., 2022; Touvron et al., 2023;  
Zhao et al., 2023)  
그 놀라운 능력으로 인해 자연어처리(NLP) 시스템에서 기초적인 구성 요소가 되었다.

LMs가 훈련 과정에서 일부 세계 지식을 암기하였음에도  
(Petroni et al., 2019; Roberts et al., 2020;  
Jiang et al., 2020),  
여전히 환각을 일으키고 허구의 내용을 만들어내는 경향이 있다  
(Maynez et al., 2020; Zhou et al., 2021).

외부 지식 자원으로부터 관련 정보를 조회하는 검색 구성 요소로 LMs를 보완하는 것은  
환각 문제를 해결하기 위한 유망한 방향이다  
(Khandelwal et al., 2020; Izacard et al., 2022).

---

검색 기반 증강 LMs는 일반적으로 사용자의 입력에 기반하여 문서를 검색하고,  
그 검색된 문서들에 조건부로 완전한 답변을 생성하는  
retrieve-and-generate 구성을 사용한다  
(Chen et al., 2017; Guu et al., 2020;  
Lewis et al., 2020; Izacard and Grave, 2021;  
Sachan et al., 2021; Lee et al., 2021;  
Jiang et al., 2022; Izacard et al., 2022;  
Nakano et al., 2021; Qian et al., 2023;  
Lazaridou et al., 2022; Shi et al., 2023).

이러한 단일 시점(single-time) 검색 기반 증강 LMs는  
순수한 파라메트릭 LMs보다 우수한 성능을 보이는데,  
특히 사실 기반 질의응답(QA)과 같이 단문 지식 집약적 생성 과제에서 그러하다  
(Kwiatkowski et al., 2019; Joshi et al., 2017).

이러한 과제에서는 사용자 입력에서 정보 요구가 명확하며,  
입력에 기반하여 단 한 번 관련 지식을 검색하는 것만으로 충분하다.

---

점점 더 강력해지는 대규모 LMs는  
장문(long-form) 출력을 생성하는 보다 복잡한 과제들에서도 능력을 보여왔다.  

예를 들어 장문 QA  
(Fan et al., 2019; Stelmakh et al., 2022),  
오픈 도메인 요약  
(Cohen et al., 2021; Hayashi et al., 2021; Giorgi et al., 2022),  
그리고 연쇄적 사고(chain-of-thought; CoT) 추론  
(Wei et al., 2022; Ho et al., 2020; Geva et al., 2021; Hendrycks et al., 2020) 등이 있다.

단문(short-form) 생성과 달리 장문 생성은  
입력만으로는 명확하지 않은 복잡한 정보 요구를 수반한다.  

사람이 논문, 에세이, 책과 같은 콘텐츠를 작성할 때 점진적으로 정보를 수집하는 것처럼,  
LM을 이용한 장문 생성 또한 생성 과정 전반에 걸쳐  
여러 조각의 지식을 수집하는 것이 필요하다.

예를 들어 특정 주제에 대한 요약문을 생성할 때,  
주제 이름(e.g., Joe Biden)에 기반한 초기 검색은  
모든 측면과 세부 내용을 포괄하지 못할 수 있다.

생성 과정에서 필요한 경우 추가 정보를 검색하는 것이 매우 중요하다.  

예를 들어 특정 측면(e.g., Joe Biden의 교육 이력)을 생성할 때나,  
특정 세부 사항(e.g., Joe Biden의 대통령 선거 출마 발표 날짜)을 생성할 때 그러하다.

---

생성 과정 전반에 걸쳐 여러 차례 검색하려는 여러 시도가 이루어져 왔다.    

이러한 시도들에는 과거 문맥을 수동적으로 사용하여  
고정된 간격으로 추가 정보를 검색하는 방식  
(Khandelwal et al., 2020; Borgeaud et al., 2022;  
Ram et al., 2023; Trivedi et al., 2022)이 포함되는데,  

이 방식은 LMs가 향후 무엇을 생성하려 하는지를 정확하게 반영하지 못하거나  
부적절한 지점에서 검색을 수행할 수 있다.

멀티홉(multihop) QA 연구들 중 일부는 전체 질문을 하위 질문들로 분해하고,  
각 하위 질문을 사용하여 추가 정보를 검색하는 방식을 사용한다  
(Press et al., 2022; Yao et al., 2022;  
Khot et al., 2022; Khattab et al., 2022).

---

우리는 다음의 질문을 던진다:  
단순하고 일반적인 검색 기반 증강 LM을 만들 수 있는가?  

즉, 생성 과정 전반에 걸쳐  
언제 무엇을 검색해야 하는지를 능동적으로 결정하며,  
다양한 장문 생성 과제에 적용될 수 있는가?

우리는 능동 검색 기반 생성(active retrieval augmented generation)에 대한  
일반화된 관점을 제시한다.  

우리가 세운 ‘언제 검색할 것인가’에 대한 가설은 다음과 같다:  
LMs는 필요한 지식을 갖추지 못한 경우에만 정보를 검색해야 한다.  

이는 수동적 검색 기반 LMs에서 발생하는 불필요하거나 부적절한 검색  
(Khandelwal et al., 2020; Borgeaud et al., 2022;  
Ram et al., 2023; Trivedi et al., 2022)을 피하기 위함이다.  

대규모 LMs는 일반적으로 잘 보정되어 있으며(well-calibrated),  
낮은 확률/확신(confidence)은 지식 부족을 나타낸다는 관찰  
(Kadavath et al., 2022)에 기반하여,  

우리는 LMs가 낮은 확률의 토큰을 생성할 때만  
검색을 수행하는 능동적 검색 전략을 채택한다.

무엇을 검색할 것인가(what to retrieve)를 결정할 때는  
LMs가 미래에 무엇을 생성하려 하는지를 고려하는 것이 중요하다.  

능동 검색의 목표는 미래 생성에 이익을 주는 것이기 때문이다.

따라서 우리는 임시 다음 문장(a temporary next sentence)을 생성함으로써  
미래를 예측하고, 이를 관련 문서를 검색하기 위한 쿼리로 사용한 뒤,  
검색된 문서들에 조건부로 다음 문장을 다시 생성하는 방식을 제안한다.

이 두 가지 측면을 결합하여  
우리는 Forward-Looking Active REtrieval augmented generation(FLARE)를 제안한다  
(그림 1 참고).

FLARE는 임시 다음 문장을 반복적으로 생성하고,  
그 문장이 낮은 확률의 토큰을 포함할 경우 이를 관련 문서 검색을 위한 쿼리로 사용하며,  
문장의 끝에 도달할 때까지 다음 문장을 재생성하는 방식으로 동작한다.

---

**그림 1:** 전방향 능동 검색 기반 생성(forward-looking active retrieval augmented generation) (FLARE)의 예시.  

사용자 입력 x와 초기 검색 결과 $D_x$으로부터 시작하여,  
FLARE는 반복적으로 임시 다음 문장을 생성한다 (회색 이탤릭체로 표시됨).

그리고 그 문장이 낮은 확률의 토큰을 포함하는지 (밑줄로 표시됨) 확인한다.  

만약 그렇다면(2단계와 3단계),  
시스템은 관련 문서들을 검색하고 그 문장을 다시 생성한다.

<img src="/assets/img/paper/activerag/image_1.png" alt="image" width="800px">  

---

FLARE는 추가적인 훈련 없이 추론 단계에서 어떤 기존 LMs에도 적용 가능하다.

GPT-3.5(Ouyang et al., 2022)가 다양한 과제에서 보여준 인상적인 성능을 고려하여,  
우리는 text-davinci-003에서 우리 방법의 효과를 검증한다.

우리는 FLARE를 장문 출력을 생성하는 네 가지 다양한 과제/데이터셋에서 평가한다.  

이에는 멀티홉 QA(2WikiMultihopQA), 상식 추론(StrategyQA), 장문 QA(ASQA),  
오픈 도메인 요약(WikiAsp) (Ho et al., 2020; Geva et al., 2021;  
Stelmakh et al., 2022; Hayashi et al., 2021)이 포함된다.

모든 과제에서 FLARE는 단일 시점(single-time) 및  
다중 시점(multi-time) 검색 기반 베이스라인들과 비교하여  
우수하거나 경쟁력 있는 성능을 달성하였으며,  
이는 우리 방법의 효과성과 일반화 성능을 입증한다.

---

## 2 검색 기반 생성 (Retrieval Augmented Generation)

우리는 단일 시점(single-time) 검색 기반 생성(retrieval augmented generation)을  
형식적으로 정의하고, 능동 검색 기반 생성(active retrieval augmented generation)의  
프레임워크를 제안한다.

### 2.1 표기법과 정의 (Notations and Definitions)

사용자 입력 $x$와 문서 코퍼스 $D = \lbrace d_i \rbrace_{i=1}^{|D|}$  
(예: 전체 위키피디아 문서들)이 주어졌을 때,  
검색 기반 증강 LMs의 목표는 다음의 답변을 생성하는 것이다:

$$
y = [s_1, s_2, \ldots, s_m] = [w_1, w_2, \ldots, w_n]
$$

이는 코퍼스에서 검색된 정보를 활용하여  
$m$개의 문장 또는 $n$개의 토큰으로 이루어진 답변을 생성하는 것이다.

---

검색 기반 LM에서는, LM이 일반적으로 검색기(retriever)와 함께 동작하며,  
검색기는 쿼리 $q$에 대해 문서 목록 $D_q = \text{ret}(q)$ 를 검색할 수 있다.  

LM은 사용자 입력 $x$와 검색된 문서들 $D_q$ 둘 모두에 조건부로 답변을 생성한다.

우리는 언제 그리고 무엇을 검색할 것인지 결정하는 여러 방법들을 살펴보는 데 초점을 두므로,  
기존 연구들 (Ram et al., 2023; Trivedi et al., 2022)을 따라  
검색된 문서들을 사용자 입력 앞에 덧붙여 향후 생성에 도움이 되도록 한다.  

이 설정은 베이스라인과 우리 방법 모두에 공정한 비교를 제공한다.

$$
y = \text{LM}([\;D_q,\; x\;]),
$$

여기서 $[\;, \;]$는 주어진 순서를 따르는 연결(concatenation)을 의미한다.

---

### 2.2 단일 시점 검색 기반 생성 (Single-time Retrieval Augmented Generation)

가장 일반적인 선택은 사용자 입력을 그대로 검색을 위한 쿼리로 사용하고,  
전체 답변을 한 번에 생성하는 것이다:

$$
y = \text{LM}([\;D_x,\; x\;]).
$$

---

### 2.3 능동 검색 기반 생성 (Active Retrieval Augmented Generation)

장문 생성을 검색을 통해 보조하기 위해,  
우리는 능동 검색 기반 생성(active retrieval augmented generation)을 제안한다.  

이는 생성 과정 전반에 걸쳐 언제 그리고 무엇을 검색할 것인지  
능동적으로 결정하는 일반적(generic) 프레임워크이다.

이 과정은 검색과 생성이 서로 교차(interleaving)되도록 한다.

형식적으로, 단계 $t$ ($t \ge 1$)에서  
검색 쿼리 $q_t$는 사용자 입력 $x$와 이전에 생성된 출력  
$y_{<t} = [y_0, \ldots, y_{t-1}]$  
둘 모두에 기반하여 구성된다:

$$
q_t = \text{qry}(x, y_{<t}),
$$

여기서 $\text{qry}(\cdot)$는 쿼리 생성(query formulation) 함수이다.

초기 단계($t = 1$)에서 이전 생성은 비어 있으며 ($y_{<1} = \varnothing$),  
사용자 입력이 초기 쿼리로 사용된다 ($q_1 = x$).

검색된 문서들 $D_{q_t}$이 주어지면, LM은 다음 검색이 발생하거나  
출력이 끝에 도달할 때까지 지속적으로 답변을 생성한다:

$$
y_t = \text{LM}([\;D_{q_t},\; x,\; y_{<t}\;]),
$$

여기서 $y_t$는 현재 단계 $t$에서 생성된 토큰을 의미한다.

LM의 입력은 검색된 문서들 $D_{q_t}$,  
사용자 입력 $x$, 그리고 이전 생성 $y_{<t}$의 연결(concatenation)이다.

우리는 이전에 검색된 문서들 $\bigcup_{t' < t} D_{q_{t'}}$을 버리고,  
현재 단계의 검색 문서들만을 사용하여  
다음 생성이 LM의 입력 길이 제한에 도달하지 않도록 한다.

---

## 3 FLARE: 전방향 능동 검색 기반 생성  
(FLARE: Forward-Looking Active REtrieval Augmented Generation)

우리의 직관은 다음과 같다.  

(1) LMs는 불필요하거나 부적절한 검색을 피하기 위해  
필요한 지식을 갖추지 못했을 때에만 정보를 검색해야 한다.  

그리고 (2) 검색 쿼리는 미래 생성의 의도를 반영해야 한다.

우리는 능동 검색 기반 생성 프레임워크를 구현하기 위해  
두 가지 전방향 능동 검색 기반 생성(FLARE) 방법을 제안한다.

첫 번째 방법은 답변을 생성하는 동안 필요할 때  
LM이 검색 쿼리를 생성하도록 프롬프트하며, 검색을 유도하는 지시문을 사용한다.  

이 방법을 $ \text{FLARE}_{\text{instruct}} $ 라고 한다.

두 번째 방법은 LM의 생성을 직접 검색 쿼리로 사용하는 방식이며,  
$ \text{FLARE}_{\text{direct}} $ 라고 한다.  

이 방법은 다음 문장을 반복적으로 생성하여 미래 주제에 대한 통찰을 얻고,  
불확실한 토큰이 존재할 경우 관련 문서들을 검색하여 다음 문장을 다시 생성한다.

---

### 3.1 검색 지시문을 활용한 FLARE  
(FLARE with Retrieval Instructions)

Toolformer(Schick et al., 2023)에서 영감을 받아,  
검색이 필요함을 표현하는 가장 직관적인 방법 중 하나는  
추가 정보가 필요할 때 “[Search(query)]”를 생성하는 것이다  
(Schick et al., 2023).

예를 들어,  
“The colors on the flag of Ghana have the following meanings.  
Red is for [Search(Ghana flag red meaning)] the blood of martyrs, …”  
와 같은 방식이다.

GPT-3.5 모델이 API 접근만을 제공하는 상황에서는,  
이러한 동작을 few-shot 프롬프트(Brown et al., 2020)를 통해 유도한다.

---

구체적으로, 다운스트림(downstream) 작업의 경우  
검색 관련 지시문과 예시들을 skill 1로서 프롬프트의 시작 부분에 두고,  
그 뒤에 다운스트림 작업의 지시문과 예시들을 skill 2로서 배치한다.  

테스트 사례가 주어지면, 우리는 LMs에게 skill 1과 skill 2를 결합하여  
작업을 수행하는 동안 검색 쿼리를 생성하도록 요구한다.

프롬프트의 구조는 Prompt 3.1에 제시되어 있으며,  
전체 상세 내용은 Prompt D.3에서 확인할 수 있다.

---

<img src="/assets/img/paper/activerag/image_2.png" alt="image" width="480px">  

---

그림 2에서 보인 바와 같이, LM이 “[Search(query)]” (회색 이탤릭체로 표시됨)을 생성하면,  
우리는 생성을 중단하고 쿼리 용어를 사용하여 관련 문서들을 검색한다.  

검색된 문서들은 다음 검색 쿼리가 생성되거나 출력이 끝에 도달할 때까지  
미래 생성을 보조하기 위해 사용자 입력 앞에 덧붙여진다.

추가 구현 상세는 부록 Appendix A에 포함되어 있다.

---

**그림 2:** 검색 지시문을 사용한 전방-예측(active) 검색 보강 생성($ \text{FLARE}_{\text{instruct}} $)의 예시.  
미래 생성을 돕기 위해 관련 정보를 검색하기 위한 검색 쿼리들을  
(회색 이탤릭체로 표시된) 반복적으로 생성한다.

<img src="/assets/img/paper/activerag/image_3.png" alt="image" width="480px">  

---

### 3.2 직접 FLARE (Direct FLARE)

우리가 블랙박스 LMs를 미세 조정할 수 없기 때문에,  
검색 지시문들을 통해 $ \text{FLARE}_{\text{instruct}} $가 생성한 질의들은  
신뢰할 수 없을 수도 있다는 것을 발견했다.  

따라서, 우리는 다음 문장을 사용하여  
언제 그리고 무엇을 검색할지를 결정하는  
보다 직접적인 방식의 순방향(active) 검색을 제안한다.

---

### 3.2.1 신뢰도 기반 능동 검색 (Confidence-based Active Retrieval)

그림 1에 나타난 바와 같이, 단계 $t$에서 우리는 먼저 검색된 문서들에 조건화하지 않고  
임시 다음 문장 $\hat{s}\_t = \text{LM}([\;x,\; y\_{<t}\;])$을 생성한다.  

그 다음 우리는 검색을 촉발할지 여부를 결정하고  
$\hat{s}\_t$를 기반으로 질의들을 공식화한다.  

만약 LM이 $\hat{s}\_t$에 대해 확신(confident)한다면,  
추가 정보를 검색하지 않고 이를 받아들인다.  

그렇지 않다면, 우리는 $\hat{s}\_t$를 사용하여  
적절한 문서를 검색하기 위한 검색 질의 $q\_t$를 만들고,  
그 다음 문장 $s\_t$를 다시 생성한다(regenerate).  

우리가 반복(iteration)의 기반으로 문장을 사용하는 이유는,  
구(phrase)처럼 너무 짧지도 않고  
단락(paragraph)처럼 너무 길지도 않은  
의미 단위(semantic units)로서 문장이 갖는 중요성 때문이다.  

그러나 우리의 접근법은 단락이나 구를 기반으로 사용할 수도 있다.

---

LM들은 잘 보정되어 있는 경향이 있기 때문에,  
낮은 확률/신뢰도는 종종 지식 부족을 나타낸다  
(Jiang et al., 2021; Kadavath et al., 2022; Varshney et al., 2022).  

우리는 $\hat{s}\_t$의 어떤 토큰이라도  
임계값 $\theta \in [0,1]$보다 낮은 확률을 가지면 능동적으로 검색을 촉발한다.  

$\theta = 0$이면 검색이 전혀 촉발되지 않음을 의미하며,  
$\theta = 1$이면 모든 문장에서 검색이 촉발된다.

$$
y_t =
\begin{cases}
\hat{s}_t & \text{만약 } \hat{s}_t \text{의 모든 토큰 확률이 } \ge \theta \text{이면} \\
s_t = \text{LM}([\; D_{q_t},\; x,\; y_{<t} \;]) & \text{그 외의 경우}
\end{cases}
$$

여기서 질의 $q\_t$는 $\hat{s}\_t$에 기초하여 공식화된다.

---

### 3.2.2 신뢰도 기반 질의 공식화 (Confidence-based Query Formulation)

검색을 수행하는 한 가지 방법은  
다음 문장 $\hat{s}\_t$를 질의 $q\_t$로 직접 사용하는 것이다.  

이는 LM이 생성한 가설적 제목(hypothetical titles)이나  
단락(paragraphs)을 검색 질의나 증거로 사용하는  
기존 방법들과 유사한 취지를 공유한다  
(Gao et al., 2022; Sun et al., 2022; Yu et al., 2022; Mao et al., 2021).  

우리는 이러한 기법들을 능동적 정보 접근이 필수적인 장문(long-form) 생성으로 일반화한다.

---

우리는 다음 문장을 사용하여 검색을 수행하는 것이  
이전 문맥을 사용하는 것보다 훨씬 더 좋은 결과를 낸다는 것을 발견했으며,  
이는 뒤의 하위절 6.2에서 보여진다.  

그러나 이는 그 안에 포함된 오류를 지속시킬 위험이 있다.  

예를 들어, LM이 “Joe Biden attended the University of Pennsylvania”  
라는 문장을 생성했으나,  
실제 사실은 그가 University of Delaware에 다녔던 것이라면,  
이 잘못된 문장을 질의로 사용하는 것은 오도하는 정보를 검색하게 만들 수 있다.  

우리는 그림 3에 나타난 바와 같이  
이 문제를 극복하기 위한 두 가지 간단한 방법을 제안한다.

---

**그림 3:** 암시적 질의 공식화와 명시적 질의 공식화.  
낮은 확률을 가진 토큰들은 밑줄로 표시된다.

<img src="/assets/img/paper/activerag/image_4.png" alt="image" width="600px">  

---

**마스킹된 문장을 암시적 질의로 사용하기**  

첫 번째 방법은 $\hat{s}\_t$에서 신뢰도가 낮은 토큰들을  
확률이 임계값 $\beta \in [0,1]$ 아래일 때 마스킹한다.  

더 높은 $\beta$는 더 공격적인 마스킹을 초래한다.  

이 방법은 문장에서 잠재적인 방해 요소들을 제거하여 검색 정확도를 향상시킨다.


---

**명시적 질의로서 생성된 질문들**

또 다른 방법은 $\hat{s}\_t$에서 신뢰도가 낮은 구간(span)을  
직접적으로 겨냥하는 명시적 질문들을 생성하는 것이다.  

예를 들어, LM이 “the University of Pennsylvania”에 대해 확신하지 못하는 경우,  
“Which university did Joe Biden attend?”와 같은 질문은  
관련 정보를 검색하는 데 도움이 될 수 있다.  

Self-ask(Press et al., 2022)는 후속 질문(follow-up questions)을  
다운스트림 작업 예시들에 수동으로 삽입함으로써 이를 달성했으며,  
이는 뒤에 Prompt D.2에서 보이듯 작업별 주석(annotation)이 필요하다.  

우리는 이러한 추가 주석 없이  
신뢰도가 낮은 구간들을 대상으로 하는 질문을 생성하는 범용적 접근법을 개발하였다.  

구체적으로, 우리는 먼저 $\hat{s}\_t$에서  
확률이 $\beta$ 아래인 모든 구간(span)들을 추출한다.  

각 추출된 구간 $z$에 대해,  
그 구간으로 대답될 수 있는 질문 $q\_{t,z}$를 생성하도록 gpt-3.5-turbo에 프롬프트한다.

---

<img src="/assets/img/paper/activerag/image_5.png" alt="image" width="480px">  

---

우리는 생성된 각 질문을 사용하여 검색을 수행하고,  
반환된 문서들을 미래 생성에 도움이 되도록  
하나의 랭킹 리스트로 끼워 넣는다(interleave).  

요약하면, 질의 $q_t$는 다음과 같이 $\hat{s}_t$를 기반으로 구성된다:

$$
q_t =
\begin{cases}
\varnothing
& \text{만약 } \hat{s}_t \text{의 모든 토큰의 확률이 } \ge \theta \text{ 인 경우} \\
\text{mask}(\hat{s}_t) \text{ 또는 } \text{qgen}(\hat{s}_t)
& \text{그 외의 경우}
\end{cases}
$$

---

## 3.3  구현 세부 사항

**Base LM**  

우리는 가장 발전된 GPT-3.5 LM 중 하나인 *text-davinci-003*을  
그들의 API를 반복적으로 질의(querying)함으로써  
우리의 방법을 검증한다.<sup>2</sup>

> <sup>2</sup> https://api.openai.com/v1/completions April 23.

**문서 코퍼스와 검색기(retrievers)**  

우리는 검색과 생성의 통합에 초점을 맞추므로,  
쿼리를 입력으로 받아 관련 문서 목록을 반환하는  
오프더셸프(off-the-shelf) 검색기를 사용한다.  

> 오프더셸프(off-the-shelf)란?  
> 
> 오프더셸프는 “선반에서 바로 가져다 쓰는”이라는 의미로,  
> 추가적인 맞춤 개발 없이 즉시 사용할 수 있는  
> 기존의 완성형 도구·모델·소프트웨어를 의미한다.  
> 즉, 사용자가 별도로 훈련하거나 설계하지 않고  
> 바로 적용할 수 있는 범용 솔루션을 말한다.

주로 Wikipedia의 지식에 의존하는 데이터셋의 경우,  
Karpukhin et al. (2020)의 Wikipedia 덤프를 사용하고  
BM25(Robertson and Zaragoza, 2009)를 검색기로 사용한다.  

오픈 웹의 지식에 의존하는 데이터셋의 경우,  
우리는 Bing 검색 엔진<sup>3</sup>을 검색기로 사용한다.  

> <sup>3</sup> https://www.microsoft.com/en-us/bing/apis/bing-web-search-api

**검색된 문서 포맷팅**  

여러 개의 검색된 문서들은 그들의 순위에 따라 일렬로 정리(linearized)된 뒤,  
Prompt D.1을 사용하여 사용자 입력의 맨 앞에 추가된다.
 
문장 토크나이제이션(sentence-tokenization) 및 효율성과 같은 기타 구현 세부 사항들은  
Appendix A에 포함되어 있다.

---

## 4  다중 시점 검색 베이스라인들 (Multi-time Retrieval Baselines)

기존의 수동적 다중 시점 검색 증강(passive multi-time retrieval augmented) LMs 역시  
우리의 프레임워크(2.3절)를 사용하여 공식화될 수 있다.  

이 절에서 우리는 언제 검색하고 무엇을 검색할지에 기반하여  
세 가지 베이스라인 범주를 공식적으로 소개한다.  

이 베이스라인들은 해당 논문의 정확한 재현이 아니다.  
이는 많은 설계 선택들이 다르기 때문에 직접적인 비교가 불가능하게 만들기 때문이다.  

우리는 동일한 설정을 사용하여 이를 구현했으며,  
유일한 차이는 언제 검색하고 무엇을 검색하는지이다.

---

**이전-윈도우(Previous-window)** 접근법은  
윈도우 크기를 나타내는 $l$ 토큰마다 한 번씩 검색을 트리거한다.  

이전 윈도우에서 생성된 토큰들이 쿼리로 사용된다:

$$
\begin{aligned}
q_t &= y_{t-1} \quad (t \ge 2), \\
y_t &= [w_{(t-1)l+1}, \ldots, w_{tl}] .
\end{aligned}
$$

이 범주에 속하는 기존 기법들로는 RETRO(Borgeaud et al., 2022),  
몇 개의 토큰마다 검색을 수행하는 IC-RALM(Ram et al., 2023),  
그리고 모든 토큰마다 검색하는 KNN-LM(Khandelwal et al., 2020)<sup>4</sup>이 있다.  

> <sup>4</sup> KNN-LM은 현재 디코딩 위치에 대응하는  
> 문맥화된 표현(contextualized representation)을 사용하여  
> 이전 모든 토큰들을 인코딩한 관련 정보를 검색한다.  
> 
> 따라서 엄밀히 말하면 $q_t$는 $y_{<t}$가 되어야 한다.  

Ram et al. (2023)을 따라 윈도우 크기 $l = 16$을 사용한다.

---

**이전-문장(previous-sentence)** 접근법은  
매 문장에서 검색을 트리거하고, 이전 문장을 쿼리로 사용하며,  
IRCoT(Trivedi et al., 2022)이 여기에 속한다:

$$
\begin{aligned}
q_t &= y_{t-1} \quad (t \ge 2), \\
y_t &= s_t .
\end{aligned}
$$

---

**질문 분해(Question decomposition)** 접근법은  
작업별 예시들을 수동으로 주석(annotation) 처리하여  
LM이 출력을 생성하는 동안 분해된 하위 질문(sub-questions)을 생성하도록 유도한다.  

예를 들어, self-ask(Press et al., 2022)는 이 범주에 속하는 방법으로,  
Prompt D.2에서 보이듯 예시(exemplar) 안에 하위 질문들을 수동으로 삽입한다.  

테스트 사례(test case)에서는  
모델이 하위 질문을 생성할 때마다 동적으로 검색이 트리거된다.  

---

앞서 언급된 접근법들은 생성 중에 추가 정보를 검색해 올 수 있다.  

그러나 이러한 방법들은 다음과 같은 뚜렷한 단점들을 가진다:  

(1) 이전에 생성된 토큰들을 쿼리로 사용하는 것은  
LM이 미래에 생성하려는 내용을 반영하지 못할 수 있다.  

(2) 고정된 간격으로 정보를 검색하는 방식은  
부적절한 지점에서 검색이 발생할 수 있어 비효율적일 수 있다.  

(3) 질문 분해(question decomposition) 기반 접근법은  
작업(task)별 프롬프트 엔지니어링이 필수적이므로,  
새로운 작업에 대한 일반화 성능을 제한한다.

---

## 5 실험 설정 (Experimental Setup)

우리는 FLARE의 효과를 few-shot in-context 학습  
(Radford et al., 2019; Brown et al., 2020; Liu et al., 2023)을 사용하여  
4개의 다양한 지식-집약적 작업에서 평가한다.  

우리는 이전 연구(Trivedi et al., 2022)를 따라  
실험 비용 때문에 각 데이터셋에서 최대 500개의 예시만 부분 샘플링한다.  

데이터셋, 측정 지표, 그리고 설정은 부록 B의 표 7에 요약되어 있다.  

FLARE의 하이퍼파라미터는 개발 세트를 기준으로 선택되며  
표 9에 나열되어 있다.  

특별히 명시되지 않은 경우, FLARE는 $ \text{FLARE}_{\text{direct}} $를 의미한다.
