---
layout: post
title: "[1-2] Probability and Statistics for Engineering and the Sciences"
date: 2025-10-07 19:00:00 +0900
categories:
  - "Books"
  - "Probability and Statistics for Engineering and the Sciences"
tags: []
---

# 1. 개요와 기술통계 (Overview and Descriptive Statistics)

---

## 1.2 기술통계에서의 그림과 표를 통한 표현 (Pictorial and Tabular Methods in Descriptive Statistics)

**기술통계(descriptive statistics)** 는 일반적으로  
두 가지 큰 영역으로 나눌 수 있다.  

이 절에서는  
데이터 집합(data set)을 **시각적 기법(visual techniques)** 으로 표현하는 방법을 다룬다.  
이후 **1.3절**과 **1.4절**에서는  
데이터 집합을 요약하는 **수치적 요약값(numerical summary measures)** 들을 살펴볼 것이다.  

여러분이 이미 익숙할 수 있는 시각적 표현 기법에는  
다음과 같은 것들이 있다.  

- **도수표(frequency table)**  
- **눈금표(tally sheet)**  
- **히스토그램(histogram)**  
- **원형 그래프(pie chart)**  
- **막대 그래프(bar graph)**  
- **산점도(scatter diagram)**  

이 절에서는  
이들 중에서도 특히 **확률(probability)** 과 **추론통계(inferential statistics)** 에서  
가장 **유용하고 관련성 높은 기법들**에 초점을 맞추어 다룬다.  

---

### 표기법 (Notation)

일반적인 **표기법(notation)** 을 미리 정의해 두면,  
다양한 실제 문제(practical problems)에  
우리의 통계적 방법과 공식을 더 쉽게 적용할 수 있다.  

하나의 표본(sample)에 포함된 **관측값(observations)** 의 개수,  
즉 **표본 크기(sample size)** 는 보통 $n$ 으로 표시한다.  

예를 들어, 다음 두 표본에 대해  
각각의 표본 크기는 모두 $n = 4$ 이다.  

- 대학교 이름의 표본:  
  $\{ \text{Stanford}, \text{Iowa State}, \text{Wyoming}, \text{Rochester} \}$  
- pH 측정값의 표본:  
  $\{ 6.3, 6.2, 5.9, 6.5 \}$  

두 개의 표본이 동시에 고려될 때에는  
각 표본의 크기를 $m$과 $n$, 또는 $n_1$과 $n_2$로 구분하여 나타낸다.  

예를 들어,  
두 종류의 디젤 엔진(diesel engines)에 대해  
열효율(thermal efficiency)을 측정한 결과가 다음과 같다고 하자.  

- 첫 번째 엔진 유형: $\{ 29.7,\ 31.6,\ 30.9 \}$  
- 두 번째 엔진 유형: $\{ 28.7,\ 29.5,\ 29.4,\ 30.3 \}$  

이 경우,  
첫 번째 표본의 크기는 $m = 3$,  
두 번째 표본의 크기는 $n = 4$ 이다.  

---

어떤 변수 $x$ 에 대한 $n$개의 관측값으로 이루어진 데이터 집합(data set)이 주어졌다고 하자.  
이 개별 관측값들은 다음과 같이 표기한다.  

$$
x_1,\ x_2,\ x_3,\ \dots,\ x_n
$$

여기서 **첨자(subscript)** 는  
각 관측값의 **크기(magnitude)** 와는 아무런 관련이 없다.  

즉,  
- $x_1$ 이 반드시 가장 작은 값일 필요는 없으며,  
- $x_n$ 이 반드시 가장 큰 값일 필요도 없다.  

많은 응용 사례에서,  
$x_1$ 은 연구자가 **처음으로 관측한 값(first observation)**,  
$x_2$ 는 **두 번째 관측값(second observation)**,  
… 이런 식으로 순차적으로 표기된다.  

따라서, 데이터 집합 내에서  
**$i$번째 관측값** 은 일반적으로 **$x_i$** 로 나타낸다.  

---

### 줄기-잎 그림 (Stem-and-Leaf Displays)

각 관측값 $x_1, x_2, \dots, x_n$ 이 최소 두 자리 숫자로 이루어진  
**수치형 데이터 집합(numerical data set)** 을 생각해 보자.  
이러한 데이터 집합을 빠르고 직관적으로 시각화하기 위한 한 가지 방법은  
**줄기-잎 그림(stem-and-leaf display)** 을 구성하는 것이다.  

---

> #### 줄기-잎 그림 구성 방법 (Constructing a Stem-and-Leaf Display)
>
> 1. **줄기(stem)** 값으로 사용할 한 자리 또는 여러 자리의 **선행 숫자(leading digits)** 를 선택한다.  
>    나머지 **뒤쪽 숫자(trailing digits)** 들은 **잎(leaf)** 으로 사용한다.  
>
> 2. 가능한 **줄기 값(stem values)** 을 세로 열(vertical column)에 나열한다.  
>
> 3. 각 관측값에 대해, 해당하는 줄기 옆에 **잎(leaf)** 을 기록한다.  
>
> 4. 줄기와 잎의 **단위(units)** 가 무엇인지 그림 어딘가에 명시한다.  

---

만약 데이터 집합이 **0에서 100 사이의 시험 점수(exam scores)** 로 구성되어 있다면,  
점수 83은 **줄기(stem)** 가 8, **잎(leaf)** 이 3이 된다.  

한편, 데이터가 **자동차 연료 효율(automobile fuel efficiencies)** 을 나타내며  
모든 값이 8.1에서 47.8 사이(mpg 단위)라면,  
**십의 자리(tens digit)** 를 줄기로 사용할 수 있다.  
예를 들어, 값 32.6은 **줄기 3**, **잎 2.6** 이 된다.  

일반적으로,  
**줄기 개수(stem count)** 가 **5개에서 20개 사이** 가 되도록 구성하는 것이 권장된다.  

---

**예제 1.6 (Example 1.6)**  

대학생들의 **음주(alcohol use)** 문제는  
단지 학계(academic community)뿐 아니라,  
그로 인한 **건강(health)** 및 **안전(safety)** 문제로 인해  
**사회 전반(society at large)** 에서도 심각한 관심을 받고 있다.  

*Journal of the American Medical Association* (JAMA, 1994: 1672–1677)에 실린  
**“Health and Behavioral Consequences of Binge Drinking in College”** 라는 논문에서는  
미국 전역의 대학 캠퍼스에서 이루어진  
**과도한 음주(heavy drinking)** 에 대한  
종합적인 연구 결과(comprehensive study)를 보고하였다.  

이 연구에서 **폭음 에피소드(binge episode)** 는  
- 남학생의 경우 **연속 5잔 이상**,  
- 여학생의 경우 **연속 4잔 이상**  
의 음주로 정의되었다.  

**그림 1.4(Figure 1.4)** 는  
**전국 대학 학부생(undergraduate students)** 중  
**폭음자(binge drinkers)** 의 **비율(percentage)** 을 나타내는  
140개의 값으로 구성된 변수  

$$
x = \text{percentage of undergraduate students who are binge drinkers}
$$  

의 **줄기-잎 그림(stem-and-leaf display)** 을 보여준다.  

(이 값들은 논문에 직접 제시되지는 않았지만,  
본 그림은 논문에 실린 데이터 요약 시각화와 일치한다.)  

---

**그림 1.4**  

*140개 대학 각각에서 폭음자(binge drinkers)의 비율(percentage)에 대한 줄기-잎 그림(stem-and-leaf display)*  

<img src="/assets/img/books/prob-stat-eng/1/image_4.png" alt="image" width="720px"> 

---

줄기-잎 그림(stem-and-leaf display)에서  
**줄기(stem)** 가 2이고 **잎(leaf)** 이 1인 첫 번째 값은  
표본에 포함된 어떤 대학의 학생 중 **21%가 폭음자(binge drinkers)** 임을 의미한다.  

만약 그림에  
**줄기 숫자(stem digits)** 와 **잎 숫자(leaf digits)** 가  
어떤 단위를 나타내는지 명시되어 있지 않다면,  
줄기 2와 잎 1로 표시된 관측값이  
**21%** 인지, **2.1%** 인지, 혹은 **0.21%** 인지를  
구분할 수 없게 된다.  

줄기-잎 그림(stem-and-leaf display)을 **직접 손으로 작성할 때(by hand)**,  
각 줄(line)에서 **잎(leaf)** 을 **작은 값에서 큰 값 순서로 정렬(ordering)** 하는 일은  
상당히 **시간이 많이 걸리는 작업(time-consuming)** 이 될 수 있다.  

하지만 이러한 정렬은  
**추가적인 정보(extra information)** 를 거의 제공하지 않거나,  
있더라도 그 효과가 미미하다.  

예를 들어,  
관측값(observations)이 학교 이름의 **알파벳 순서(alphabetical order)** 로 정렬되어  
다음과 같이 제시되었다고 하자.  

$$
16\%,\ 33\%,\ 64\%,\ 37\%,\ 31\%,\ \dots
$$  

이 경우에도  
줄기-잎 그림을 통해 얻을 수 있는 통계적 요약 정보(summary information)는  
크게 달라지지 않는다.  

그런 다음 이러한 값들을 이 순서대로 그림에 배치하면,  
줄기 1 행의 첫 번째 잎은 6이 되고,  
줄기 3 행의 시작 부분은 다음과 같이 된다.  

3 $\mid$ 371 ...

이 줄기-잎 그림(display)은  
**전형적이거나 대표적인 값(typical or representative value)** 이  
**줄기 4 행(stem 4 row)**, 즉 **약 40%대 중반(mid-40%) 범위** 에 있음을 시사한다.  

관측값들은 이 대표값 주변에 매우 밀집되어 있지는 않다.  
예를 들어, 모든 값이 20%에서 49% 사이에 있다면  
더 높은 집중도(high concentration)를 보였을 것이다.  

줄기-잎 그림의 형태는  
아래로 내려가면서 점차 상승하여 **하나의 봉우리(single peak)** 를 형성한 뒤,  
다시 하강하는 **단봉형(unimodal)** 분포를 보인다.  
또한, **공백(gaps)** 은 존재하지 않는다.  

그림의 모양은 완벽한 **대칭(symmetric)** 은 아니며,  
**큰 값(high leaves)** 방향보다  
**작은 값(low leaves)** 방향으로 약간 더 길게 늘어진 형태를 보인다.  

마지막으로,  
데이터의 대부분으로부터 **비정상적으로 멀리 떨어진 관측값(outlier)** 은 없다.  
예를 들어, 26% 값 중 하나가 86%였다면  
이 값은 명백한 이상치(outlier)가 되었을 것이다.  

이 데이터의 가장 놀라운 점은,  
표본에 포함된 대부분의 대학에서  
**학생의 최소 25% 이상이 폭음자(binge drinkers)** 라는 사실이다.  

즉,  
**대학 캠퍼스 내 과도한 음주(heavy drinking)** 문제는  
많은 사람들이 생각했던 것보다 훨씬 **광범위하고 심각한(pervasive and serious)** 문제임을 보여준다.  

---

줄기-잎 그림(stem-and-leaf display)은  
데이터의 다음과 같은 측면(aspects)에 대한 정보를 전달한다.  

- **전형적이거나 대표적인 값(typical or representative value)** 의 확인  
- **대표값 주변의 분산 정도(extent of spread)**  
- **데이터 내 공백(gaps)** 의 존재 여부  
- **값들의 분포(distribution)** 가 보이는 **대칭성(symmetry)** 의 정도  
- **봉우리(peaks)** 의 개수 및 위치(location)  
- **이상치(outlying values)** 의 존재 여부  

---

**예제 1.7 (Example 1.7)**  

**그림 1.5(Figure 1.5)** 는  
미국에서 가장 **도전적인 골프 코스(challenging golf courses)** 로  
*Golf Magazine* 에 의해 선정된 코스들 중  
무작위로 표본 추출된 **골프장 길이(lengths of golf courses, 단위: yard)** 의  
줄기-잎 그림(stem-and-leaf displays)을 보여준다.  

---

**그림 1.5**  

*골프 코스 길이(golf course lengths)에 대한 줄기-잎 그림(stem-and-leaf displays)*  

(a) **두 자리 잎(two-digit leaves)** 를 사용한 줄기-잎 그림  
(b) **한 자리 잎(one-digit leaves)** 으로 단축(truncated)된 **Minitab 출력(display from Minitab)**  

<img src="/assets/img/books/prob-stat-eng/1/image_5.png" alt="image" width="800px"> 

---

이 40개 코스 표본(sample) 중  
가장 짧은 코스는 **6433야드**,  
가장 긴 코스는 **7280야드** 이다.  

코스 길이 값들은  
표본 내 구간 전반에 걸쳐 **대체로 고르게 분포(roughly uniform fashion)** 되어 있는 것으로 보인다.  

여기서 **줄기(stem)** 를 한 자리 숫자(예: 6 또는 7)로 선택하면  
줄기 개수가 너무 적어 정보가 부족하게 되고,  
반대로 세 자리 숫자(예: 643, …, 728)를 줄기로 선택하면  
줄기 개수가 너무 많아져서 복잡해진다.  

따라서 이 경우,  
**적절한 줄기 선택(stem choice)** 이  
정보를 효과적으로 전달하기 위해 매우 중요하다.  

---

일반적으로 **통계 소프트웨어(statistical software)** 패키지는  
**여러 자리의 줄기(multiple-digit stems)** 를 사용하는  
줄기-잎 그림(stem-and-leaf display)을 생성하지 않는다.  

**그림 1.5(b)** 의 **Minitab 출력(display)** 은  
각 관측값(observation)의 **일의 자리(ones digit)** 를 **잘라내어(truncate)**  
표현한 결과이다.  

---

### 점도표 (Dotplots)

**점도표(dotplot)** 는  
데이터의 크기가 비교적 작거나,  
서로 다른 **고유한 값(distinct data values)** 이 많지 않을 때  
수치형 데이터를 요약하기 위한 시각적으로 매력적인 방법이다.  

각 관측값(observation)은  
수평 측정축(horizontal measurement scale)의 해당 위치 위에  
하나의 **점(dot)** 으로 표시된다.  

같은 값이 여러 번 등장하면,  
그 횟수만큼 점을 **수직으로 쌓아 올린다(stack vertically)**.  

줄기-잎 그림(stem-and-leaf display)과 마찬가지로,  
점도표(dotplot)는 다음과 같은 정보를 제공한다.  

- 데이터의 **대표적 위치(location)**  
- 값들의 **산포 정도(spread)**  
- **극단값(extremes)** 의 존재 여부  
- **빈 구간(gaps)** 의 존재 여부  

---

**예제 1.8 (Example 1.8)**  

다음은 **2006–2007 회계연도(fiscal year)** 에 대한  
**주별 고등교육 예산 비율(appropriations for higher education as a percentage of state and local tax revenue)** 데이터이다.  
출처는 *Statistical Abstract of the United States* 이며,  
값들은 주의 약어(state abbreviations)에 따라  
알라바마(AL)부터 와이오밍(WY) 순으로 나열되어 있다.  

| 10.8 | 6.9 | 8.0 | 8.8 | 7.3 | 3.6 | 4.1 | 6.0 | 4.4 | 8.3 |
| 8.1 | 8.0 | 5.9 | 5.9 | 7.6 | 8.9 | 8.5 | 8.1 | 4.2 | 5.7 |
| 4.0 | 6.7 | 5.8 | 9.9 | 5.6 | 5.8 | 9.3 | 6.2 | 2.5 | 4.5 |
| 12.8 | 3.5 | 10.0 | 9.1 | 5.0 | 8.1 | 5.3 | 3.9 | 4.0 | 8.0 |
| 7.4 | 7.5 | 8.4 | 8.3 | 2.6 | 5.1 | 6.0 | 7.0 | 6.5 | 10.3 |

**그림 1.6(Figure 1.6)** 은  
이 데이터를 이용해 작성된 **점도표(dotplot)** 를 보여준다.  

가장 눈에 띄는 특징은  
주별(state-to-state)로 나타나는 **큰 변동성(substantial variability)** 이다.  

가장 큰 값(뉴멕시코, New Mexico)과  
가장 작은 두 값(뉴햄프셔, New Hampshire / 버몬트, Vermont)은  
데이터의 주된 분포(bulk of the data)로부터 다소 떨어져 있지만,  
그 차이가 **이상치(outlier)** 로 간주될 만큼 크지는 않다.  

---

**그림 1.6**  

*예제 1.8의 데이터에 대한 점도표*  

<img src="/assets/img/books/prob-stat-eng/1/image_6.png" alt="image" width="720px"> 

---

만약 **예제 1.2** 에서의 **압축 강도(compressive strength)** 관측값의 개수 $n=27$ 이  
실제로 얻어진 것보다 훨씬 더 많았다면,  
**점도표(dotplot)** 를 작성하는 것은 매우 번거로운 일이 되었을 것이다.  

이제 소개할 **다음 기법(next technique)** 은  
이러한 상황에 특히 적합하다.  

---

### 히스토그램 (Histograms)

일부 **수치형 데이터(numerical data)** 는  
변수(variable)의 값을 **세어(counting)** 얻는다.  
예를 들어,  
- 지난 1년 동안 한 사람이 받은 **교통 위반 딱지의 수(number of traffic citations)**,  
- 특정 기간 동안 서비스 센터에 도착한 **고객 수(number of customers)**  
등이 이에 해당한다.  

반면 다른 데이터들은 **측정(measurement)** 을 통해 얻어진다.  
예를 들어,  
- 개인의 **체중(weight)**,  
- 특정 자극(stimulus)에 대한 **반응 시간(reaction time)**  
등이 이에 해당한다.  

이 두 경우(세는 데이터 vs. 측정 데이터)에 따라  
**히스토그램(histogram)** 을 그리는 방법(prescription for drawing)이  
일반적으로 다르게 적용된다.  
