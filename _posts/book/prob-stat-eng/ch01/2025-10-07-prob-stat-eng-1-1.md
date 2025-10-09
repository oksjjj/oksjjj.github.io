---
layout: post
title: "[1-1] Probability and Statistics for Engineering and the Sciences"
date: 2025-10-07 18:10:00 +0900
categories:
  - "Books"
  - "Probability and Statistics for Engineering and the Sciences"
tags: []
---

# 1. 개요 및 기술 통계 (Overview and Descriptive Statistics)

---

## 1.1 모집단, 표본, 그리고 과정 (Populations, Samples, and Processes)

공학자와 과학자들은 직업적인 영역이든 일상적인 활동이든,  
항상 **사실(facts)** 혹은 **데이터(data)** 의 집합에 노출되어 있다.  

**통계학(statistics)** 은 이러한 데이터를 **조직하고 요약(summary)** 하는 방법과,  
데이터에 포함된 정보를 바탕으로 **결론을 도출하는 방법**을 제공한다.  

---

하나의 연구(investigation)는 일반적으로  
명확하게 정의된 **객체들의 집합(collection of objects)** 을 대상으로 하며,  
이 집합을 **관심 모집단(population of interest)** 이라고 한다.  

예를 들어,  
- 한 연구에서는 특정 기간 동안 생산된 **특정 종류의 젤라틴 캡슐 전부**가 모집단이 될 수 있고,  
- 또 다른 연구에서는 **최근 학년도에 공학 학사(B.S. in engineering)** 를 받은  
  모든 개인들이 모집단이 될 수 있다.  

---

만약 모집단 내의 **모든 개체(object)** 에 대한  
원하는 정보가 얻어졌다면, 이를 **전수조사(census)** 라고 부른다.  

하지만 시간, 비용, 혹은 기타 제한된 자원(resource)으로 인해  
전수조사를 수행하는 것은 **비현실적이거나 불가능한 경우가 많다.**  

대신, 모집단의 **일부(subset)** 를 **일정한 방법(prescribed manner)** 으로 선택하여  
이 부분집합을 **표본(sample)** 이라고 한다.  

---

예를 들어,  
- 특정 생산 공정(run)에서 생산된 **베어링(bearing)** 일부를 표본으로 뽑아  
  그것들이 제조 사양(specification)에 부합하는지를 조사할 수 있고,  
- 혹은 **작년 공학과 졸업생들 중 일부를 표본으로 선정**하여  
  공학 교육과정(curriculum)의 품질에 대한 피드백을 얻을 수도 있다.  

---

우리가 모집단(population)에 속한 객체(object)들을 연구할 때,  
그 모든 속성 전체에 관심을 가지는 것은 아니다.  

대부분의 경우, 특정한 몇 가지 **특성(characteristics)** 에만 관심이 있다.  

예를 들어,  
- 각 케이싱(casing) 표면의 **결함 개수(number of flaws)**,  
- 각 캡슐 벽의 **두께(thickness)**,  
- 공학 졸업생의 **성별(gender)**,  
- **졸업 시 연령(age)** 등이 이에 해당한다.  

특성은 **범주형(categorical)** 일 수도 있고, **수치형(numerical)** 일 수도 있다.  

전자의 경우, 특성의 값은 하나의 **범주(category)** 로 나타나며  
예를 들어, 성별의 경우 *여성(female)*,  
고장 유형의 경우 *납땜 불량(insufficient solder)* 등이 있다.  

후자의 경우, 특성의 값은 숫자로 표현된다.  
예를 들어,  
$age = 23\ \text{years}$,  
$diameter = 0.502\ \text{cm}$ 과 같은 값이 이에 해당한다.  

---

**변수(variable)** 란 모집단 내의 객체마다 값이 달라질 수 있는 특성이다.  
변수는 보통 알파벳의 뒷부분에 있는 소문자로 표기한다.  

예를 들어,  
$x =$ 학생이 소유한 계산기 브랜드(brand of calculator)  
$y =$ 특정 기간 동안의 웹사이트 방문 횟수(number of visits)  
$z =$ 주어진 조건에서 자동차의 제동거리(braking distance)$  

---

데이터(data)는 하나의 변수 혹은 두 개 이상의 변수에 대해  
관찰(observation)을 수행한 결과이다.  

단일 변수에 대한 관찰값들로 구성된 데이터를  
**단변량 자료(univariate data set)** 라고 한다.  

예를 들어,  
특정 자동차 대리점에서 최근 구입한 10대의 자동차에 대해  
변속기(transmission)가 자동(A)인지 수동(M)인지 조사했다면  
다음과 같은 범주형 단변량 자료가 된다.  

M A A A M A A M A A  

또한, 특정 용도에서 사용된 D 브랜드 배터리의  
수명(시간, hours)을 측정한 다음과 같은 데이터는  
수치형 단변량 자료가 된다.  

5.6 5.1 6.2 6.0 5.8 6.5 5.8 5.5  

---

두 개의 변수에 대한 관찰값들로 구성된 경우  
이를 **이변량 자료(bivariate data)** 라고 한다.  

예를 들어, 농구팀의 각 선수에 대해  
키(height)와 몸무게(weight)를 함께 기록한 데이터는 다음과 같다.  

(72, 168), (75, 212), …  

또는,  
공학자가 부품의 수명($x =$ component lifetime)과  
고장 원인($y =$ reason for failure)을 함께 기록한다면,  
이는 하나의 수치형 변수와 하나의 범주형 변수를 포함하는  
이변량 데이터 세트가 된다.  

---

두 개 이상의 변수에 대한 관찰이 이루어진 경우에는  
**다변량 자료(multivariate data)** 가 된다.  
(이변량은 다변량의 특수한 경우이다.)  

예를 들어, 한 연구 의사가 각 환자에 대해  
수축기 혈압(systolic blood pressure),  
이완기 혈압(diastolic blood pressure),  
혈청 콜레스테롤 수치(serum cholesterol level)를 측정한 경우,  
각 관찰값은 (120, 80, 146)과 같은 세 개의 수치로 이루어진다.  

많은 다변량 자료에서는 일부 변수는 수치형이고  
다른 일부는 범주형일 수 있다.  

예를 들어, *Consumer Reports*의 연간 자동차 특집호에는  
다음과 같은 변수들이 포함된다.  

- 차량 종류(type of vehicle): 소형(small), 스포츠형(sporty), 중형(mid-size), 대형(large)  
- 시내 연비(city fuel efficiency, mpg)  
- 고속도로 연비(highway fuel efficiency, mpg)  
- 구동 방식(drivetrain type): 전륜(front wheel), 후륜(rear wheel), 4륜(four wheel)  

이처럼 다변량 데이터는  
수치형과 범주형 변수가 함께 존재하며,  
복합적인 현상을 종합적으로 분석할 수 있도록 해준다.  

### 통계학의 분류 (Branches of Statistics)

데이터를 수집한 연구자(investigator)는  
단순히 데이터를 **요약(summarize)** 하고 **주요 특징(describe important features)** 을  
묘사하고자 할 수도 있다.  

이러한 목적에는 **기술통계(descriptive statistics)** 의 방법들이 사용된다.  

---

이러한 방법 중 일부는 **그래픽(graphical)** 성격을 가지며,  
대표적인 예는 다음과 같다.  

- **히스토그램(histogram)**  
- **상자그림(boxplot)**  
- **산점도(scatter plot)**  

다른 기술통계적 방법들은  
**수치 요약 통계량(numerical summary measures)** 의 계산을 포함한다.  

예를 들어,  
- 평균(mean)  
- 표준편차(standard deviation)  
- 상관계수(correlation coefficient)  
등이 있다.  

---

통계용 컴퓨터 소프트웨어(statistical computer software packages)의  
광범위한 보급으로 인해,  
이러한 작업들은 과거보다 훨씬 더 손쉽게 수행할 수 있게 되었다.  

컴퓨터는 인간보다 계산(calculation)이나 그래프 작성(creation of pictures)에  
훨씬 효율적이며,  
사용자로부터 올바른 명령(instructions)을 받기만 하면 된다.  

이 말은 곧, 연구자가 더 이상 **반복적이고 단순한 작업(grunt work)** 에  
시간을 허비할 필요가 없으며,  
그 시간을 **데이터를 분석하고 중요한 메시지를 추출**하는 데  
더 많이 사용할 수 있음을 의미한다.  

---

이 책 전반에 걸쳐,  
**Minitab**, **SAS**, **S-Plus**, **R** 등의 다양한 통계 패키지의  
출력(output) 예시를 함께 제시할 것이다.  

이 중 **R 소프트웨어**는 무료로 다운로드할 수 있으며,  
공식 사이트 주소는 다음과 같다.  

👉 <a href="https://www.r-project.org" target="_blank">https://www.r-project.org</a>

---

**예제 1.1 (Example 1.1)**  

미국에서는 **자선사업(charity)** 이 거대한 산업으로 자리 잡고 있다.  

웹사이트 charitynavigator.com 에는  
약 **5,500개의 자선단체(charitable organizations)** 에 대한 정보가 수록되어 있으며,  
이 외에도 그 목록에 포함되지 않은 **더 작은 규모의 단체들**이 다수 존재한다.  

일부 자선단체들은  
총지출(total expenses)에서 모금활동(fundraising)과 행정비용(administrative expenses)이  
차지하는 비중이 매우 작을 정도로 **효율적으로 운영**된다.  
반면, 어떤 단체들은 전체 지출의 상당 부분을  
이러한 활동에 사용하는 경우도 있다.  

다음은 **무작위로 선택된 60개 자선단체**에 대해  
총지출 대비 **모금비용 비율(%)** 을 조사한 데이터이다.  

|   |   |   |   |   |   |   |   |   |   |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 6.1 | 12.6 | 34.7 | 1.6 | 18.8 | 2.2 | 3.0 | 2.2 | 5.6 | 3.8 |
| 2.2 | 3.1 | 1.3 | 1.1 | 14.1 | 4.0 | 21.0 | 6.1 | 1.3 | 20.4 |
| 7.5 | 3.9 | 10.1 | 8.1 | 19.5 | 5.2 | 12.0 | 15.8 | 10.4 | 5.2 |
| 6.4 | 10.8 | 83.1 | 3.6 | 6.2 | 6.3 | 16.3 | 12.7 | 1.3 | 0.8 |
| 8.8 | 5.1 | 3.7 | 26.3 | 6.0 | 48.0 | 8.2 | 11.7 | 7.2 | 3.9 |
| 15.3 | 16.6 | 8.8 | 12.0 | 4.7 | 14.7 | 6.4 | 17.0 | 2.5 | 16.2 |


이 데이터를 아무런 정리 없이 보면,  
그 특징적인 패턴을 직관적으로 파악하기 어렵다.  

예를 들어, 다음과 같은 질문에 바로 답하기가 쉽지 않다.  

- 대표적인 값(typical or representative value)은 어느 정도인가?  
- 값들이 특정한 중심값 주변에 밀집되어 있는가, 아니면 넓게 퍼져 있는가?  
- 데이터에 **공백(gap)** 이 존재하는가?  
- **20% 미만의 비율**을 보이는 자선단체는 얼마나 되는가?  

이를 보다 명확하게 파악하기 위해,  
그림 1.1(Figure 1.1)에서는  
**줄기-잎 그림(stem-and-leaf display)** 과 **히스토그램(histogram)** 을 함께 제시하였다.  

---

**그림 1.1**  
*Minitab 줄기-잎 그림(stem-and-leaf display, 소수점 첫째 자리 절삭)과  
자선단체 모금비율 데이터의 히스토그램(histogram)*  

<img src="/assets/img/books/prob-stat-eng/1/image_1.png" alt="image" width="720px"> 

---

1.2절에서는 이러한 **데이터 요약 도표(data summaries)** 의  
작성 방법(construction)과 해석(interpretation)에 대해 논의할 것이다.  

지금은 단지,  
이러한 시각화 방법들이 **0에서 100 사이의 가능한 값 범위에서  
모금비율이 어떻게 분포하는지**를 보여주기 시작함을 이해하면 된다.  

이 표본(sample)에 포함된 자선단체들 중  
**대다수는 모금비용이 총지출의 20% 미만**이며,  
그중 일부만이 **상식적인 수준을 벗어난(outlier) 비율**을  
보이는 것으로 확인된다.  

---

모집단(population)으로부터 표본(sample)을 얻은 연구자(investigator)는  
종종 이 표본으로부터 얻은 정보를 이용하여  
모집단에 대한 어떤 형태의 결론(conclusion)을 내리고자 한다.  
즉, 표본은 그 자체가 목적이 아니라 **목적에 이르는 수단(means to an end)** 이다.  

표본으로부터 모집단으로 **일반화(generalize)** 하는 데 사용되는 기법들은  
통계학의 한 분야인 **추론통계(inferential statistics)** 에 속한다.  

---

**예제 1.2 (Example 1.2)**  

재료 강도(material strength)에 대한 연구는  
통계적 방법(statistical methods)이 광범위하게 활용되는 분야 중 하나이다.  

논문 *“Effects of Aggregates and Microfillers on the Flexural Properties of Concrete”*  
(*Magazine of Concrete Research*, 1997: 81–98) 에서는  
**고성능 콘크리트(high-performance concrete)** 의 강도 특성을 연구하였다.  
이 콘크리트는 **고성능 감수제(superplasticizer)** 와  
특정 결합제(binder)를 사용하여 제작된 것이다.  

이전에는 주로 **압축강도(compressive strength)** 가 연구되었지만,  
**굽힘강도(flexural strength)** — 즉, 휨 하중에서 파괴를 견디는 능력 —  
에 대해서는 알려진 바가 많지 않았다.  

다음은 논문에 제시된 콘크리트의 **굽힘강도(flexural strength)** 데이터이며,  
단위는 **메가파스칼(MPa)** 이다.  

$$
1\ \text{Pa (Pascal)} = 1.45 \times 10^{-4}\ \text{psi}
$$

|   |   |   |   |   |   |   |   |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 5.9 | 7.2 | 7.3 | 6.3 | 8.1 | 6.8 | 7.0 | 7.6 |
| 6.8 | 6.5 | 7.0 | 6.3 | 7.9 | 9.0 | 8.2 | 8.7 |
| 7.8 | 9.7 | 7.4 | 7.7 | 9.7 | 7.8 | 7.7 | 11.6 |
| 11.3 | 11.8 | 10.7 |   |   |   |   |   |  

이제 이러한 방식으로 제작할 수 있는 **모든 보(beam)** 의  
평균 굽힘강도(flexural strength)의 값을 추정하고자 한다고 하자.  

즉, “이와 같은 방식으로 만들어질 수 있는 모든 보의 모집단(population)”을  
가정(conceptualize)한다면,  
우리가 추정하려는 것은 바로 **모집단 평균(population mean)** 이다.  

계산 결과, 높은 신뢰 수준(high degree of confidence)에서  
모집단의 평균 굽힘강도는 다음 범위에 포함된다고 할 수 있다.

$$
7.48\ \text{MPa} \leq \mu \leq 8.80\ \text{MPa}
$$

이 구간을 **신뢰구간(confidence interval)** 또는 **구간 추정(interval estimate)** 이라고 한다.  

또한, 이 데이터를 사용하여  
이와 같은 방식으로 제작된 **단일 보(single beam)** 의  
굽힘강도를 예측할 수도 있다.  

높은 신뢰 수준에서,  
이 보의 굽힘강도는 **7.35 MPa 이상**일 것으로 예측된다.  

이때의 값 **7.35**를 **하한 예측값(lower prediction bound)** 이라고 한다.  

---

이 책의 주요 초점(main focus)은  
**과학적 연구(scientific work)** 에 유용한 **추론통계(inferential statistics)** 방법들을  
제시하고 그 활용 사례를 설명하는 데 있다.  

가장 중요한 세 가지 추론 절차(inferential procedures) —  
1. **점추정(point estimation)**,  
2. **가설검정(hypothesis testing)**,  
3. **신뢰구간에 의한 추정(estimation by confidence intervals)** —  
은 제6~8장에서 소개되며,  
제9~16장에서는 이를 **보다 복잡한 상황**에 적용한다.  

이 장(제1장)의 나머지 부분에서는  
**추론 기법의 발전에 가장 자주 사용되는 기술통계(descriptive statistics)** 방법들을 다룬다.  

---

제2~5장은 **확률(probability)** 분야의 내용을 다루며,  
이 확률 이론은 기술통계와 추론통계를 **연결해주는 다리(bridge)** 역할을 한다.  

확률을 숙달하게 되면,  
다음과 같은 점들을 보다 명확히 이해할 수 있다.  

- 추론 절차(inferential procedures)가 **어떻게 개발되고 사용되는지**,  
- 통계적 결론(statistical conclusions)을 **일상 언어로 해석하고 설명하는 방법**,  
- 그리고 **통계 방법을 적용할 때 발생할 수 있는 함정(pitfalls)** 들이  
  언제, 어디에서 나타날 수 있는지.  

확률(probability)과 통계(statistics)는  
모두 **모집단(populations)** 과 **표본(samples)** 에 관련된 문제를 다루지만,  
이 두 분야는 **서로 ‘역(inverse)적인 방식’으로 접근**한다는 점에서 다르다.  

---

확률(probability) 문제에서는  
연구 대상인 **모집단(population)** 의 특성이 이미 **주어진 것(known)** 으로 가정된다.  

예를 들어, **수치형 모집단(numerical population)** 의 경우  
모집단 값들이 어떤 **특정한 분포(distribution)** 를 따른다고 가정할 수 있다.  

이때의 목표는,  
그 모집단으로부터 추출된 **표본(sample)** 에 관한 질문을 제기하고  
그에 대한 답을 찾는 것이다.  

반면, **통계(statistics)** 문제에서는 상황이 정반대이다.  

실험자(experimenter)는 **표본의 특성(sample characteristics)** 만을 알고 있으며,  
이 표본으로부터 얻은 정보를 이용해  
**모집단에 대한 결론(conclusions about the population)** 을 도출한다.  

이 두 학문 간의 관계는 다음과 같이 요약할 수 있다.  

- **확률(probability)** 은 **모집단에서 표본으로 추론한다.**  
  → 이는 **연역적 추론(deductive reasoning)** 에 해당한다.  

- **추론통계(inferential statistics)** 는 **표본에서 모집단으로 추론한다.**  
  → 이는 **귀납적 추론(inductive reasoning)** 에 해당한다.  

이 관계는 **그림 1.2(Figure 1.2)** 에 시각적으로 표현되어 있다.  

---

**그림 1.2**  

*확률(probability)과 추론통계(inferential statistics) 간의 관계*  

<img src="/assets/img/books/prob-stat-eng/1/image_2.png" alt="image" width="480px"> 

---

특정 표본(sample)이 모집단(population)에 대해  
무엇을 알려줄 수 있는지를 이해하기 전에,  
먼저 **주어진 모집단으로부터 표본을 추출할 때 발생하는 불확실성(uncertainty)** 을  
이해해야 한다.  

바로 이러한 이유 때문에  
우리는 **통계를 배우기 전에 확률(probability)을 먼저 공부**한다.  

---

**예제 1.3 (Example 1.3)**  

확률(probability)과 추론통계(inferential statistics)의 초점이 어떻게 다른지를 보여주는 예로,  
**자동 어깨 벨트(automatic shoulder belt)** 가 장착된 차량에서의  
**수동 랩 벨트(manual lap belt)** 사용 사례를 생각해 보자.  

(참고: *“Automobile Seat Belts: Usage Patterns in Automatic Belt Systems,”*  
*Human Factors*, 1998: 126–135)  

확률의 관점에서는,  
예를 들어 특정 도시권 내의 이러한 차량 운전자 중 **50%가 정기적으로 랩 벨트를 착용한다**  
는 가정을 세운다고 하자.  
이것은 **모집단(population)에 대한 가정**이다.  

이때 다음과 같은 질문을 할 수 있다.  
- “이러한 운전자 100명을 무작위로 선택했을 때,  
  그중 **70명 이상이 정기적으로 랩 벨트를 착용할 확률**은 얼마인가?”  
- “표본 크기가 100명일 때,  
  **평균적으로 몇 명이 랩 벨트를 착용할 것으로 기대할 수 있는가?**”  

반면, 추론통계의 관점에서는 상황이 반대이다.  
이제 우리는 **표본 정보(sample information)** 를 가지고 있다.  

예를 들어,  
자동 어깨 벨트가 장착된 차량 운전자 100명을 조사한 결과,  
그중 **65명이 정기적으로 랩 벨트를 착용한다**는 사실을 알게 되었다고 하자.  

이 경우 다음과 같은 질문을 던질 수 있다.  
- “이 결과가,  
  **이 지역의 전체 운전자 중 50% 이상이 랩 벨트를 정기적으로 착용한다**  
  는 결론을 내릴 만큼 충분한 근거(substantial evidence)를 제공하는가?”  

즉, 이 두 번째 상황에서는  
**표본 정보(sample information)** 를 이용하여  
그 표본이 추출된 **전체 모집단(population)** 의 구조에 대해  
질문에 답하려는 시도를 하고 있는 것이다.  

---

앞서 다룬 **랩 벨트 예제(lap belt example)** 에서는  
모집단(population)이 명확하고 구체적으로 정의되어 있다.  
즉, **특정 도시권 내에서 일정한 방식으로 장착된 차량의 모든 운전자**가  
그 모집단에 해당한다.  

반면, **예제 1.2**의 경우는 다르다.  
그때의 **굽힘강도(strength)** 측정값들은  
기존의 모집단에서 선택된 것이 아니라,  
**시제품 보(prototype beams)** 의 표본(sample)에서 얻어진 것이다.  

이 경우에는 **모집단을 실제로 존재하는 집합으로 볼 수 없기 때문에**,  
“**유사한 실험 조건(similar experimental conditions)** 아래에서  
측정될 수 있는 **모든 가능한 강도 측정값(all possible strength measurements)**”을  
모집단으로 **개념화(conceptualize)** 하는 것이 편리하다.  

이러한 모집단을 **개념적(conceptual)** 또는 **가상적(hypothetical) 모집단** 이라고 부른다.  

이처럼 여러 문제 상황에서는,  
**모집단을 개념적으로 설정함으로써(conceptualizing a population)**  
그 질문을 **추론통계(inferential statistics)** 의 틀 안에 맞추어 다룰 수 있다.  

---

### 현대 통계학의 범위 (The Scope of Modern Statistics)

오늘날 **통계적 방법론(statistical methodology)** 은  
거의 모든 학문 분야(disciplines)에서 연구자들(investigators)에 의해 활용되고 있다.  

그 적용 분야의 예시는 다음과 같다.  

- **분자생물학(molecular biology)**  
  → 마이크로어레이 데이터(microarray data)의 분석  

- **생태학(ecology)**  
  → 다양한 동물 및 식물 모집단(animal and plant populations)의  
    공간적 분포(spatial distribution)를 정량적으로 기술  

- **재료공학(materials engineering)**  
  → 금속 부식을 억제하기 위한 다양한 처리법(treatments)의 특성 연구  

- **마케팅(marketing)**  
  → 신제품 출시를 위한 **시장조사(market survey)** 및  
    **마케팅 전략(marketing strategies)** 개발  

- **공중보건(public health)**  
  → 질병의 원인(source)과 치료 방법(treatment method)의 규명  

- **토목공학(civil engineering)**  
  → 구조 요소(structural elements)에 미치는 응력(stress)의 영향과  
    교통 흐름(traffic flow)이 지역사회에 미치는 영향을 평가  

이처럼 **현대 통계학(modern statistics)** 은  
기초과학에서 공학, 사회과학, 산업 분야에 이르기까지  
**모든 연구와 실무의 핵심 도구**로 사용되고 있다.  

---

이 책을 계속 읽어 나가다 보면,  
**확률(probability)** 과 **통계(statistics)** 의 다양한 기법들이  
실제 문제 상황에서 어떻게 적용되는지를 보여주는  
폭넓은 예시와 연습문제들을 만나게 될 것이다.  

이들 예시의 상당수는  
**공학(engineering)** 및 **과학(science)** 관련 학술지에 실린  
논문(article)들에서 발췌한 **데이터(data)** 혹은 **연구 내용(material)** 을 기반으로 한다.  

이 책에서 다루는 방법들은  
데이터를 다루는 연구자들에게 있어  
이미 **검증되고 신뢰받는 도구(established and trusted tools)** 로 자리 잡았다.  

한편, 통계학자(statisticians)들은  
여전히 **무작위성(randomness)** 과 **불확실성(uncertainty)** 을 기술하기 위한  
새로운 모델(model)을 개발하고 있으며,  
데이터 분석을 위한 **새로운 방법론(new methodology)** 역시 꾸준히 제시하고 있다.  

통계학계(statistical community)의  
이러한 **지속적인 창의적 노력(creative efforts)** 의 증거로,  
최근 통계학 저널들에 발표된 일부 논문들의  
**제목(titles)** 과 **요약 설명(capsule descriptions)** 을 아래에 소개한다.  

참고로,  
- *Journal of the American Statistical Association* 은 **JASA**,  
- *Annals of Applied Statistics* 는 **AAS** 로 약칭된다.  

이 두 저널은 통계학 분야에서 가장 저명한 학술지들 중 일부이다.  

- **“Modeling Spatiotemporal Forest Health Monitoring Data”** (*JASA*, 2009: 899–911)  
  1980년대 유럽 전역에서는  
  대기오염(air pollution)으로 인한 **산림 쇠퇴(forest dieback)** 문제에 대응하기 위해  
  **산림 건강 모니터링 시스템(forest health monitoring systems)** 이 구축되었다.  
  이후 이 시스템은 **기후 변화(climate change)** 와 **오존 농도 증가(ozone levels)** 로 인한  
  새로운 위협을 중심으로 운영이 지속되었다.  
  저자들은 나무의 **수관 탈엽(tree crown defoliation)** — 즉, 나무 건강의 지표 — 을  
  정량적으로 기술하기 위한 **통계적 모델(statistical model)** 을 개발하였다.  

- **“Active Learning Through Sequential Design, with Applications to the Detection of Money Laundering”** (*JASA*, 2009: 969–981)  
  **자금 세탁(money laundering)** 은 불법 활동으로 얻은 자금의 출처를 감추는 행위를 말한다.  
  금융기관(financial institutions)에서 매일 발생하는 방대한 거래량으로 인해  
  자금 세탁 행위를 탐지(detection)하기가 매우 어렵다.  
  기존의 접근 방식은 거래 내역(transaction history)으로부터  
  여러 요약 통계(summary quantities)를 추출하고,  
  의심스러운 활동을 **시간이 많이 드는 조사(time-consuming investigation)** 로 확인하는 것이었다.  
  이 논문에서는 훨씬 더 **효율적인 통계적 탐지 방법(efficient statistical method)** 을 제안하고,  
  실제 사례(case study)를 통해 그 효과를 보여준다.  

- **“Robust Internal Benchmarking and False Discovery Rates for Detecting Racial Bias in Police Stops”** (*JASA*, 2009: 661–668)  
  경찰의 행동(police actions)이 **인종적 편향(racial bias)** 에 기인한다는  
  의혹(allegation)은 여러 지역사회에서 논쟁의 대상이 되어 왔다.  
  이 논문은 **“거짓 양성(false positives)”**,  
  즉 **편향이 없는데도 편향이 있는 것으로 잘못 식별되는 사례**를  
  최소화하도록 설계된 **새로운 통계적 방법(new statistical method)** 을 제안한다.  

  이 방법은 2006년 **뉴욕시(New York City)** 에서 실시된  
  **50만 건의 보행자 검문(pedestrian stops)** 데이터를 분석하는 데 적용되었다.  
  그 결과, 정기적으로 검문에 참여한 **3,000명의 경찰관 중 15명**이  
  **흑인(Black)** 및 **히스패닉(Hispanic)** 보행자를  
  **편향이 없다고 가정할 경우 예상되는 비율보다 유의하게 높은 비율**로  
  검문한 것으로 확인되었다.  

- **“Records in Athletics Through Extreme Value Theory”** (*JASA*, 2008: 1382–1391)  
  이 논문은 **세계 육상 경기(world athletics)** 에서의  
  **기록(record)의 극단값(extremes)** 을 모델링하는 데 초점을 맞추고 있다.  

  저자들은 다음의 두 가지 핵심 질문을 던진다.  
  1. 특정 종목에서 **달성 가능한 궁극적인 세계기록(ultimate world record)** 은 무엇인가?  
     (예: 여성 높이뛰기 등)  
  2. 현재의 세계기록은 **얼마나 ‘우수한(good)’ 기록인지**,  
     그리고 서로 다른 종목 간에 그 **기록의 수준(quality)** 은 어떻게 비교되는가?  

  연구에서는 총 **28개 종목(남녀 각각 14개: 8개 달리기, 3개 투척, 3개 도약 종목)** 을 분석하였다.  

  예를 들어,  
  - 남자 마라톤의 경우, 현 기록에서 **약 20초 정도만 단축 가능**할 것으로 예상되며,  
  - 반면 여자 마라톤의 현재 기록은 **궁극적으로 달성 가능한 수준보다 약 5분가량 더 느리다.**  

  또한, 이 방법론은  
  **공항 활주로(runway)** 의 길이가 충분한지,  
  **네덜란드 제방(dikes in Holland)** 의 높이가 충분한지와 같은  
  극단적 위험 상황(extreme risk conditions)에 대한  
  **안전성 평가(safety assessment)** 문제에도 응용될 수 있다.  

- **“Analysis of Episodic Data with Application to Recurrent Pulmonary Exacerbations in Cystic Fibrosis Patients”** (*JASA*, 2008: 498–510)  
  이 논문은 **반복적 의학적 사건(recurrent medical events)** 의 분석에 초점을 둔다.  
  예를 들어, **편두통(migraine headache)** 과 같은 질환의 경우,  
  첫 발병 시점뿐 아니라 각 발병이 **얼마나 지속되는가(length of episode)** 도  
  중요한 정보를 제공한다.  

  이러한 **에피소드의 길이(duration)** 는  
  질병의 **심각도(severity)**, **의료비용(medical cost)**,  
  **삶의 질(quality of life)** 등에 대한 중요한 단서를 제공할 수 있다.  

  저자들은 **에피소드의 발생 빈도(frequency)** 와 **지속 시간(length)** 을  
  함께 요약할 수 있는 새로운 분석 기법을 제안하며,  
  또한 시간의 경과에 따라  
  **에피소드 발생에 영향을 주는 요인(characteristics)** 들의 효과가  
  변화할 수 있도록 모델링하였다.  

  이 방법은 **낭포성 섬유증(Cystic Fibrosis, CF)** 환자 데이터를 대상으로 적용되었다.  
  CF는 **땀샘(sweat gland)** 과 기타 **분비샘(exocrine glands)** 에 영향을 미치는  
  심각한 **유전성 질환(genetic disorder)** 이다.  

---

요즘 들어 **통계 정보(statistical information)** 는  
대중 매체(popular media)에 점점 더 자주 등장하고 있으며,  
때로는 그 초점이 **통계학자(statisticians)** 들 자신에게 향하기도 한다.  

예를 들어,  
**2009년 11월 23일자 *뉴욕타임스(The New York Times)*** 는  
*“Behind Cancer Guidelines, Quest for Data”* 라는 제목의 기사에서  
다음과 같은 내용을 보도하였다.  

새로운 **암 연구 과학(cancer science)** 과  
점점 더 정교해진 **데이터 분석 기법(sophisticated data analysis methods)** 이  
미국 **예방 서비스 실무 그룹(U.S. Preventive Services Task Force)** 으로 하여금  
**중년 및 노년 여성의 유방촬영(mammogram) 검사 주기**에 대한  
기존 지침(guidelines)을 재검토하도록 자극했다는 것이다.  

이 위원회(panel)는  
**6개의 독립적인 연구팀(independent groups)** 에게  
**통계 모델링(statistical modeling)** 을 의뢰하였으며,  
그 결과 새로운 결론들이 도출되었다.  

그 중 핵심 주장은 다음과 같다.  
- **2년에 한 번 시행하는 유방촬영검사**는  
  매년 시행하는 경우에 비해 **효과는 거의 동일하지만**,  
  **잠재적 부작용(risk of harms)** 은 절반 수준으로 낮다.  

유명한 생물통계학자(biostatistician) **도널드 베리(Donald Berry)** 는  
이 연구 결과를 인용하며  
“실무 그룹이 새로운 연구 결과를 진지하게 반영했다는 점이 놀랍고도 반갑다”고 말했다.  

그러나 이 위원회의 보고서는  
**암 관련 단체(cancer organizations)**,  
**정치인(politicians)**,  
그리고 **여성들(women themselves)** 사이에서  
큰 논란(controversy)을 불러일으켰다.  

---

이 책과 함께 통계학(statistics)을 더 깊이 탐구해 나가면서,  
여러분이 점점 더 **이 학문이 지닌 중요성과 실질적 관련성(importance and relevance)** 을  
실감하게 되기를 바란다.  

그리고 나아가,  
이 과정이 여러분이 통계학에 더욱 흥미를 느끼고(**turned on**),  
현재의 과정을 넘어 **통계학적 배움(statistical education)** 을  
계속 이어가고자 하는 계기가 되기를 희망한다.  

---

### 열거적 연구(Enumerative Studies)와 분석적 연구(Analytic Studies)

미국의 저명한 통계학자 **W. E. 데밍(W. E. Deming)** 은  
1950~1960년대 일본의 **품질 혁신(quality revolution)** 을 주도한 인물로,  
그는 통계학에서 **열거적 연구(enumerative studies)** 와 **분석적 연구(analytic studies)** 의 구분을 제시하였다.  

**열거적 연구(enumerative study)** 에서는  
관심의 초점이 **유한하고(finite), 명확하게 식별 가능하며(identifiable), 변하지 않는(unchanging)**  
특정 집합 — 즉, 모집단(population)을 구성하는 개인(individuals)이나 객체(objects) — 에 맞춰진다.  

이 경우, **표본틀(sampling frame)**,  
즉 표본 추출 대상이 되는 개인이나 객체의 목록(listing)은  
연구자가 이미 가지고 있거나, 필요시 새로 구성할 수 있다.  

예를 들어,  

- 특정 선거에서 **청원서(petition)** 에 서명한 모든 사람들의 명단이 표본틀이 될 수 있다.  
  표본은 **유효한 서명(valid signatures)** 의 수가  
  특정 기준을 초과하는지 여부를 판단하기 위해 선택될 수 있다.  

- 또 다른 예로,  
  특정 회사에서 일정 기간 동안 제조한 **난방로(furnace)** 의 **일련번호(serial number)** 목록이  
  표본틀이 될 수 있으며,  
  이 표본을 통해 해당 제품들의 **평균 수명(average lifetime)** 에 대해 추론할 수 있다.  

이와 같은 **열거적 연구의 맥락**에서는,  
이 책에서 다루게 될 **통계적 추론 기법(inferential methods)** 의 사용이  
대체로 논쟁의 여지가 없다.  
(물론, 어떤 특정 방법을 적용할 것인가에 대해서는 통계학자들 간의 의견 차이가 있을 수 있다.)  

---

**분석적 연구(Analytic Study)** 는  
그 성격상 **열거적(enumerative)** 이지 않은 연구로 넓게 정의된다.  

이러한 연구는 주로 **미래의 제품(future product)** 을 개선하기 위해  
현재의 **공정(process)** 에 어떤 조치를 취하는 것을 목적으로 수행된다.  
예를 들어,  
- 장비의 **재보정(recalibration)** 이나,  
- **촉매(catalyst)** 투입량과 같은 입력 수준의 **조정(adjustment)** 등이 그 예이다.  

이 경우, 데이터(data)는  
**현재 존재하는 공정(existing process)** 에서만 얻을 수 있으며,  
이 공정은 미래의 공정과 여러 중요한 측면에서 다를 수 있다.  
따라서 관심 있는 개체(individuals)나 물체(objects)를 나열한  
**표본틀(sampling frame)** 은 존재하지 않는다.  

예를 들어,  
새로운 설계(new design)를 적용한 **5개의 터빈(turbines)** 을  
실험적으로 제작하여 효율(efficiency)을 평가한다고 하자.  
이 5개의 터빈은  
**유사한 조건에서 제조될 수 있는 모든 시제품(prototypes)** 의  
**개념적 모집단(conceptual population)** 에서 추출된 표본으로 볼 수는 있지만,  
**정식 생산(regular production)** 이 시작된 이후 제조될 터빈 모집단을  
대표한다고 보기는 어렵다.  

따라서, 이러한 표본 정보를 이용해  
**미래 생산 단위(future production units)** 에 대해 결론을 내리는 것은  
문제가 될 수 있다.  
이러한 **외삽(extrapolation)** 이 합리적인지 판단하기 위해서는,  
해당 분야(예: 터빈 설계 및 공학)에 대한 전문 지식을 가진 사람의  
판단이 필요하다.  

이와 같은 문제들에 대한 훌륭한 논의는  
**Gerald Hahn** 과 **William Meeker** 의 논문  
*“Assumptions for Statistical Inference”*  
(*The American Statistician*, 1993: 1–11) 에 자세히 제시되어 있다.  

---

### 데이터 수집 (Collecting Data)

통계학(statistics)은  
이미 수집된 데이터(data)의 **정리(organization)** 와 **분석(analysis)** 만을 다루는 것이 아니라,  
그 데이터를 **어떻게 수집할 것인가(data collection)** 에 대한  
기법의 개발(development of techniques)도 포함한다.  

데이터가 적절히 수집되지 않으면,  
연구자(investigator)는 자신이 다루고자 하는 문제에 대해  
**합리적인 수준의 신뢰(confidence)** 를 가지고  
질문에 답할 수 없게 된다.  

데이터 수집 과정에서 흔히 발생하는 문제 중 하나는,  
**결론을 내리려는 대상 모집단(target population)** 과  
**실제로 표본이 추출된 모집단(actually sampled population)** 이  
서로 다를 수 있다는 점이다.  

예를 들어,  
광고주(advertisers)는  
잠재 고객(potential customers)의 **TV 시청 습관(viewing habits)** 에 대해  
다양한 정보를 알고 싶어 한다.  

이와 같은 체계적(systematic) 정보는  
미국 전역의 일부 가정에 **시청 모니터링 장치(monitoring devices)** 를 설치하여 얻는다.  
그러나 연구자들은  
이 장치의 설치 자체가 **시청 행동(viewing behavior)** 에 영향을 미쳐,  
결국 **표본의 특성(sample characteristics)** 이  
**대상 모집단(target population)** 의 특성과 달라질 수 있다고 추정하였다.  

---

데이터 수집(data collection)이  
표본틀(frame)로부터 개인(individuals)이나 객체(objects)를 선택하는 과정을 포함할 때,  
**대표성 있는 표본(representative selection)** 을 확보하는 가장 단순한 방법은  
**단순 무작위 표본추출(simple random sampling)** 을 사용하는 것이다.  

단순 무작위 표본이란,  
예를 들어 표본 크기가 100이라면,  
**가능한 모든 100개 표본 조합이 동일한 확률로 선택될 수 있는 표본**을 말한다.  

예를 들어,  
표본틀이 **1,000,000개의 일련번호(serial numbers)** 로 구성되어 있다면,  
1부터 1,000,000까지의 번호를 각각  
동일한 종이조각에 적어 상자(box)에 넣고,  
충분히 섞은 뒤 하나씩 꺼내어  
원하는 표본 크기만큼 뽑을 수 있다.  

그러나 실제로는 이러한 수작업 방식보다는,  
**난수표(table of random numbers)** 나  
**컴퓨터의 난수 생성기(random number generator)** 를 이용하는 것이  
훨씬 효율적이며 일반적으로 권장된다.  

---

경우에 따라서는,  
표본을 단순 무작위로 추출하는 대신  
**표본 추출 과정을 더 용이하게 하거나(selection process easier)**,  
**추가 정보를 얻거나(extra information)**,  
**결론의 신뢰 수준(confidence in conclusions)** 을 높이기 위해  
다른 표본추출 방법(alternative sampling methods)을 사용할 수도 있다.  

그중 하나가 **층화표본추출(stratified sampling)** 이다.  

층화표본추출이란,  
모집단(population)을 **서로 겹치지 않는(nonoverlapping)** 여러 **집단(strata)** 으로 구분하고,  
각 층(stratum)에서 **별도의 표본(sample)** 을 추출하는 방법이다.  

예를 들어,  
한 **DVD 플레이어 제조업체(manufacturer)** 가  
지난 1년 동안 생산된 제품에 대한 **고객 만족도(customer satisfaction)** 를 알고자 한다고 하자.  

만약 그 기간 동안 **3가지 모델(models)** 이 생산·판매되었다면,  
각 모델에 해당하는 층(stratum)에서 **별도의 표본**을 추출할 수 있다.  

이 방법을 사용하면  
세 모델 각각에 대한 정보를 모두 얻을 수 있으며,  
특정 모델이 전체 표본에서 **과대표집(overrepresented)** 되거나  
**과소대표집(underrepresented)** 되는 문제를 방지할 수 있다.  

---

종종 **편의표본(convenience sample)** 은  
체계적인 무작위화(systematic randomization) 없이  
개인(individuals)이나 객체(objects)를 **편의상 선택(conveniently selected)** 하여 얻어진다.  

예를 들어,  
벽돌(bricks) 더미에서 표본을 추출한다고 하자.  
만약 벽돌이 쌓인 구조상 **중앙의 벽돌은 꺼내기 어렵고**,  
**위쪽이나 옆쪽의 벽돌만 선택할 수 있다면**,  
추출된 표본은 모집단 전체를 대표하지 못할 수 있다.  

즉,  
만약 쌓인 벽돌의 위치(상단·측면·중앙)에 따라  
벽돌의 특성이 다르다면,  
결과적으로 얻어진 표본 데이터(sample data)는  
**대표성(representativeness)** 을 잃게 된다.  

연구자(investigator)는 종종  
이러한 편의표본이 **무작위표본(random sample)** 에  
가까운 근사값(approximation)을 제공한다고 가정하고,  
그 위에서 **통계적 추론(inferential methods)** 을 수행하려 한다.  
그러나 이는 **연구자의 판단(judgment call)** 에 의존하는 부분이다.  

이 책에서 다루는 대부분의 통계적 방법들은  
제5장에서 설명할 **단순 무작위표본추출(simple random sampling)** 의  
변형된 형태를 기반으로 한다.  

---

공학자(engineers)와 과학자(scientists)들은  
종종 **설계된 실험(designed experiment)** 을 수행하여 데이터를 수집한다.  

이러한 실험은 여러 가지 **처리(treatments)** —  
예를 들어 **비료(fertilizer)**, **부식 방지 코팅(coating for corrosion protection)** 등 — 를  
여러 **실험 단위(experimental units)** —  
예: **토지 구획(plots of land)**, **파이프 조각(pieces of pipe)** — 에  
**어떻게 배정할 것인가(allocation)** 를 결정하는 과정을 포함할 수 있다.  

또는, 연구자(investigator)는  
특정 요인(factors)의 수준(levels)이나 범주(categories)를  
체계적으로 변화시키고(systematically vary),  
그 결과가 **반응변수(response variable)** 에 미치는 영향을 관찰할 수도 있다.  

예를 들어,  
- **압력(pressure)** 이나  
- **절연 재료의 종류(type of insulating material)**  
을 변화시키면서,  
**생산 공정에서의 산출량(yield)** 이 어떻게 변하는지를 살펴보는 것이다.  

---

**예제 1.4 (Example 1.4)**  

1987년 1월 27일자 *뉴욕타임스(The New York Times)* 에 실린 한 기사에서는  
**아스피린(aspirin)** 복용이 **심장마비(heart attack)** 위험을 줄일 수 있다는  
연구 결과가 보도되었다.  

이 결론은 **설계된 실험(designed experiment)** 에 근거한 것으로,  
두 그룹을 대상으로 비교가 이루어졌다.  

- **대조군(control group)** :  
  아스피린과 외형은 같지만 실제로는 약효가 없는  
  **위약(placebo)** 을 복용한 그룹  
- **처치군(treatment group)** :  
  정해진 복용 지침(specified regimen)에 따라  
  **아스피린** 을 복용한 그룹  

연구 참여자(subjects)들은  
**편향(bias)** 을 방지하고,  
**확률 기반(probability-based)** 분석 방법을 적용할 수 있도록  
**무작위로(randomly assigned)** 두 그룹에 배정되었다.  

실험 결과는 다음과 같다.  

| 구분 | 대상자 수 | 심장마비 발생자 수 |
|:--:|:--:|:--:|
| 대조군 (placebo) | 11,034명 | 189명 |
| 처치군 (aspirin) | 11,037명 | 104명 |

처치군에서의 **심장마비 발생률(incidence rate)** 은  
대조군의 약 **절반 수준(about half)** 에 불과했다.  

이러한 결과에 대한 한 가지 가능한 설명은  
**우연 변동(chance variation)** 일 수 있다.  
즉, 실제로는 아스피린이 효과가 없지만,  
두 동전(two identical coins)을 던졌을 때  
앞면이 나오는 횟수가 달라지는 것처럼  
관찰된 차이가 단순한 **표본의 우연한 변동(typical sample variation)**  
일 수 있다는 것이다.  

그러나 이 경우,  
**통계적 추론 방법(inferential methods)** 은  
단순한 우연 변동만으로는  
이러한 큰 차이(magnitude of the observed difference)를  
**충분히 설명할 수 없음을 시사한다.**  

---

**예제 1.5 (Example 1.5)**  

한 엔지니어(engineer)가  
특정 기판(substrate)에 **집적회로(IC, integrated circuit)** 를 장착할 때,  
**접착제의 종류(adhesive type)** 와 **도체 재료(conductor material)** 가  
**결합 강도(bond strength)** 에 미치는 영향을 조사하고자 한다.  

두 가지 접착제와 두 가지 도체 재료가 고려되고 있으며,  
각 조합(adhesive-type / conductor-material)마다  
두 번의 측정(observations)이 이루어졌다.  

그 결과 데이터는 다음과 같다.  

| 접착제 종류<br>(Adhesive Type) | 도체 재료<br>(Conductor Material) | 관측된 결합 강도<br>(Observed Bond Strength) | 평균<br>(Average) |
|:--:|:--:|:--:|:--:|
| 1 | 1 | 82, 77 | 79.5 |
| 1 | 2 | 75, 87 | 81.0 |
| 2 | 1 | 84, 80 | 82.0 |
| 2 | 2 | 78, 90 | 84.0 |

이 데이터를 기반으로 한 평균 결합 강도는 **그림 1.3(Figure 1.3)** 에 시각화되어 있다.  

---

**그림 1.3**  

*예제 1.5의 평균 결합 강도 (Average Bond Strengths in Example 1.5)*  

<img src="/assets/img/books/prob-stat-eng/1/image_3.png" alt="image" width="480px"> 

---

이를 보면,  
**접착제 2번(adhesive type 2)** 은  
어떤 도체 재료를 사용하든 **접착제 1번보다 결합 강도가 높으며**,  
**(2, 2)** 조합이 가장 높은 결합 강도를 보인다.  

이러한 차이가 실제로 존재하는 효과(real effect)인지,  
아니면 단순한 **우연 변동(chance variation)** 에 의한 것인지를 판단하기 위해  
다시 한 번 **통계적 추론 방법(inferential methods)** 을 사용할 수 있다.  

추가로,  
만약 여기에 두 가지 **경화 시간(cure time)** 과  
두 가지 **IC 포스트 코팅(post coating)** 유형이 고려된다면,  
총 가능한 조합의 수는 다음과 같다.

$$
2 \times 2 \times 2 \times 2 = 16
$$

이 경우,  
엔지니어는 **모든 조합에 대해 단 하나의 측정조차 수행하기 어려울** 수 있다.  

**11장(Chapter 11)** 에서는  
이와 같은 상황에서 **가능한 조합의 일부만 신중하게 선택(fractional selection)** 하여도  
필요한 정보를 효율적으로 얻을 수 있는 방법을 다룬다.  