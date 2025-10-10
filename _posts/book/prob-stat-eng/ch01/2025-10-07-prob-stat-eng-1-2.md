---
layout: post
title: "[1-2] 공학 및 과학을 위한 확률과 통계"
date: 2025-10-07 19:00:00 +0900
categories:
  - "Books"
  - "Probability and Statistics for Engineering and the Sciences"
tags: []
---

# 1. 개요 및 기술통계(Descriptive Statistics)  

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

---

>**정의 (DEFINITION)**  
>
>어떤 **수치형 변수(numerical variable)** 가 가질 수 있는 값들의 집합이  
>유한(finite)하거나, 혹은  
>**순서 있게 나열 가능한 무한 수열(infinite sequence)** —  
>즉, 첫 번째 값, 두 번째 값, 세 번째 값처럼  
>차례로 셀 수 있는 경우 — 이면  
>그 변수를 **이산형(discrete)** 변수라 한다.  
>
>반면,  
>가능한 값들이 **수직선(number line)** 위의 **하나의 구간(entire interval)** 전체를 이루는 경우,  
>그 변수를 **연속형(continuous)** 변수라 한다.  

---

이산형 변수(discrete variable) $x$ 는  
대부분의 경우 **세는 과정(counting)** 에서 발생한다.  

이때 가능한 값들은  
0, 1, 2, 3, … 과 같은 **정수들의 부분집합(subset of integers)** 이 된다.  

반면 **연속형 변수(continuous variable)** 는  
**측정(measurement)** 을 통해 얻어진다.  

예를 들어,  
$x$ 가 어떤 화학 물질의 **pH 값** 을 의미한다고 하자.  
이 경우, 이론적으로 $x$ 는  
0과 14 사이의 어떤 수라도 될 수 있다 —  
즉 7.0, 7.03, 7.032 등 무한히 많은 값이 가능하다.  

물론 실제로는  
모든 **측정 기기(measuring instrument)** 가  
정확도(accuracy)에 한계를 가지므로,  
pH, 반응 시간(reaction time), 키(height), 농도(concentration) 등의 값을  
임의의 자릿수까지 측정할 수는 없다.  

그러나  
**데이터의 분포(distribution of data)** 를 표현하기 위한  
**수학적 모델(mathematical model)** 을 만들 때에는  
이 변수가 **연속적인 값의 범위(entire continuum of possible values)** 를 가진다고  
가정하는 것이 유용하다.  

---

이산형 변수(discrete variable) $x$ 에 대한 관측값들로 이루어진 데이터를 생각해보자.  

어떤 특정한 $x$ 값이 데이터 집합에서 나타나는 횟수를  
그 값의 **도수(frequency)** 라고 한다.  

어떤 값의 **상대도수(relative frequency)** 는  
그 값이 전체 데이터에서 차지하는 **비율(fraction or proportion)** 을 의미한다.  

$$
\text{값의 상대도수} =
\frac{\text{그 값이 나타난 횟수}}
{\text{데이터 집합의 전체 관측값 수}}
$$

예를 들어,  
어떤 데이터 집합이 $x =$ “한 대학생이 이번 학기에 수강하는 과목 수”  
에 대한 **200개의 관측값(observations)** 으로 구성되어 있다고 하자.  

그중 70개의 $x$ 값이 3이라면,  

- **값 3의 도수(frequency of the x value 3)** : 70  

- **값 3의 상대도수(relative frequency of the x value 3)** :  

$$
\frac{70}{200} = 0.35
$$

---

상대도수(relative frequency)에 100을 곱하면 **백분율(percentage)** 이 된다.  

예를 들어, 앞선 대학 과목 수 예시에서  
표본에 포함된 학생 중 **35%가 세 과목을 수강 중**임을 의미한다.  

일반적으로,  
**도수(frequency)** 그 자체보다  
**상대도수(relative frequency)** 혹은 **백분율(percentage)** 이  
더 큰 관심의 대상이 된다.  

이론적으로는,  
모든 상대도수의 합은 1이 되어야 한다.  
그러나 실제 계산에서는 **반올림(rounding)** 때문에  
그 합이 1에서 약간 벗어날 수 있다.  

**도수분포표(frequency distribution)** 란  
각 값에 대한 **도수(frequency)** 와/또는 **상대도수(relative frequency)** 를  
표 형태로 정리한 것을 말한다.  

---

>**이산형 데이터(discrete data)에 대한 히스토그램 작성법**
>
>1. 먼저, 각 $x$ 값에 대한 **도수(frequency)** 와 **상대도수(relative frequency)** 를 계산한다.  
>
>2. 가능한 $x$ 값들을 **수평축(horizontal scale)** 위에 표시한다.  
>
>3. 각 $x$ 값 위에  
>   그 값의 **상대도수(또는 도수)** 에 해당하는 **높이(height)** 의 **직사각형(rectangle)** 을 그린다.  
>
>이렇게 만들어진 그림이 **이산형 변수에 대한 히스토그램(histogram for discrete data)** 이다.  

---

이러한 방식으로 구성하면,  
각 **직사각형의 면적(area)** 이 해당 값의 **상대도수(relative frequency)** 에 비례하도록 보장된다.  

예를 들어,  
$x = 1$ 의 상대도수가 0.35이고  
$x = 5$ 의 상대도수가 0.07이라면,  

$x = 1$ 위의 직사각형의 면적은  
$x = 5$ 위의 직사각형 면적의 **5배** 가 된다.  

---

**예제 1.9 (Example 1.9)**  

메이저리그 야구(Major League Baseball) 경기에서  
**노히트(no-hitter)** 또는 **원히트(one-hitter)** 가 얼마나 드문 일일까?  
또한, 한 팀이 한 경기에서 **10개, 15개, 혹은 20개 이상의 안타(hits)** 를  
기록하는 일은 얼마나 자주 일어날까?  

**표 1.1(Table 1.1)** 은  
**1989년부터 1993년까지** 진행된  
모든 **9이닝 경기(all nine-inning games)** 에 대해  
한 팀이 경기당 기록한 **안타 수(number of hits per team per game)** 에 대한  
**도수분포표(frequency distribution)** 를 보여준다. 

---

**표 1.1**  

9이닝 경기에서의 안타 수에 대한 도수분포표  

<img src="/assets/img/books/prob-stat-eng/1/image_7.png" alt="image" width="720px"> 

---

**그림 1.7(Figure 1.7)** 의 히스토그램은  
완만하게(single peak) 하나의 꼭짓점으로 상승한 뒤 다시 감소한다.  

또한, 히스토그램의 오른쪽(큰 값 쪽) 꼬리가  
왼쪽보다 조금 더 길게 뻗어 있으며,  
이는 **약한 양의 왜도(slight positive skew)** 를 나타낸다.  

---

**그림 1.7**

9이닝 경기당 안타 수의 히스토그램  

<img src="/assets/img/books/prob-stat-eng/1/image_8.png" alt="image" width="600px"> 

---

도수분포표(tabulated information) 또는 히스토그램(histogram)으로부터  
다음과 같은 사실을 알 수 있다.  

**두 개 이하의 안타(at most two hits)** 를 기록한 경기의 비율은  

$$
\text{두 개 이하의 안타를 기록한 경기의 비율}
$$

$$
= \text{(x = 0)의 상대도수}  +
\text{(x = 1)의 상대도수}  +
\text{(x = 2)의 상대도수} 
$$

$$
= 0.0010 + 0.0037 + 0.0108 = 0.0155
$$

비슷하게,  
**5개 이상 10개 이하의 안타** 를 기록한 경기의 비율은  

$$
\text{5개 이상 10개 이하의 안타를 기록한 경기의 비율} =
0.0752 + 0.1026 + \cdots + 0.1015 = 0.6361
$$

즉, 전체 경기의 약 **64%** 가  
**5개에서 10개(포함)** 사이의 안타를 기록하였다.  

---

연속형 데이터(continuous data, 측정값)에 대한 히스토그램을 작성하는 것은  
측정 축(measurement axis)을 적절한 개수의 **계급구간(class intervals)** 또는 **계급(classes)** 으로  
나누는 과정을 포함한다.  

각 관측값(observation)은 정확히 **하나의 계급** 에 포함되어야 한다.  

예를 들어,  
$x =$ 연료 효율(fuel efficiency, mpg)에 대한  
50개의 관측값이 있다고 하자.  

이 중 가장 작은 값은 27.8이고,  
가장 큰 값은 31.4이다.  

그렇다면 다음과 같이 계급 경계(class boundaries)를 설정할 수 있다.  

<img src="/assets/img/books/prob-stat-eng/1/image_9.png" alt="image" width="600px"> 

---

하나의 잠재적 어려움은,  
어떤 관측값이 **계급 경계(class boundary)** 위에 놓이는 경우가 있다는 점이다.  
이 경우 해당 관측값은 정확히 하나의 구간에 속하지 않게 된다.  
예를 들어, 값이 29.0인 경우가 그렇다.  

이 문제를 해결하는 한 가지 방법은  
다음과 같이 **소수 둘째 자리(hundredths digit)** 를 추가하여  
계급 경계를 설정하는 것이다.  

예: 27.55, 28.05, … , 31.55  

이렇게 하면 관측값이 경계선 위에 위치하는 일을 방지할 수 있다.  

또 다른 방법은  
다음과 같은 구간을 사용하는 것이다.  

27.5– < 28.0, 28.0– < 28.5, … , 31.0– < 31.5  

이 경우 29.0은  
28.5– < 29.0 구간이 아닌 **29.0– < 29.5 구간** 에 포함된다.  

즉,  
이 규칙(convention)에 따르면,  
**경계 위에 위치한 관측값은 항상 그 오른쪽 구간(interval to the right)** 에 포함된다.  

이 방법은 **Minitab** 이 히스토그램을 구성할 때 사용하는 방식이다.  

---

>**연속형 데이터에 대한 히스토그램 작성법: 동일한 계급 폭**  
>
>1. 각 **계급(class)** 에 대한 **도수(frequency)** 와 **상대도수(relative frequency)** 를 구한다.  
>
>2. **계급 경계(class boundaries)** 를 **수평 측정축(horizontal measurement axis)** 에 표시한다.  
>
>3. 각 **계급 구간(class interval)** 위에,  
>   그에 대응하는 **상대도수(또는 도수)** 를 높이로 하는 **직사각형(rectangle)** 을 그린다.  

---

**예제 1.10 (Example 1.10)**  

전력 회사(power companies)는  
정확한 **수요 예측(forecasts of demands)** 을 위해  
고객의 **에너지 사용량(customer usage)** 에 대한 정보를 필요로 한다.  

위스콘신 파워 앤 라이트(Wisconsin Power and Light)의 연구자들은  
특정 기간 동안 **가스 난방 주택(gas-heated homes)** 90가구를 표본으로 하여  
**에너지 소비량(energy consumption, BTUs)** 을 측정하였다.  

조정된 소비량(adjusted consumption)은 다음과 같이 계산되었다.  

$$
\text{조정된 소비량} =
\frac{\text{소비량}}
{(\text{날씨(일 단위 온도: in degree days)})(\text{주택 면적})}
$$

그 결과는 Minitab에 저장된 데이터 세트 **FURNACE.MTW** 의 일부로 포함되어 있으며,  
그 데이터를 **가장 작은 값에서 가장 큰 값 순서로 정렬**하였다.  

---

다음은 조정된 소비량(adjusted consumption)의 관측값이다.  

| 2.97 | 4.00 | 5.20 | 5.56 | 5.94 | 5.98 | 6.35 | 6.62 | 6.72 | 6.78 |
| 6.80 | 6.85 | 6.94 | 7.15 | 7.16 | 7.23 | 7.29 | 7.62 | 7.62 | 7.69 |
| 7.73 | 7.87 | 7.93 | 8.00 | 8.26 | 8.29 | 8.37 | 8.47 | 8.54 | 8.58 |
| 8.61 | 8.67 | 8.69 | 8.81 | 9.07 | 9.27 | 9.37 | 9.43 | 9.52 | 9.58 |
| 9.60 | 9.76 | 9.82 | 9.83 | 9.83 | 9.84 | 9.96 | 10.04 | 10.21 | 10.28 |
| 10.28 | 10.30 | 10.35 | 10.36 | 10.40 | 10.49 | 10.50 | 10.64 | 10.95 | 11.09 |
| 11.12 | 11.21 | 11.29 | 11.43 | 11.62 | 11.70 | 11.70 | 12.16 | 12.19 | 12.28 |
| 12.31 | 12.62 | 12.69 | 12.71 | 12.91 | 12.92 | 13.11 | 13.38 | 13.42 | 13.43 |
| 13.47 | 13.60 | 13.96 | 14.24 | 14.35 | 15.12 | 15.24 | 16.06 | 16.90 | 18.26 |

---

**Minitab** 에게 계급 구간(class intervals)을 자동으로 선택하게 하였다.  

**그림 1.8(Figure 1.8)** 의 히스토그램에서  
가장 두드러진 특징은 그것이 **종 모양(bell-shaped)**,  
즉 **대칭적(symmetric)** 인 곡선을 닮았다는 점이다.  

대칭의 중심점(point of symmetry)은  
대략 **10 근처**에 위치한다.  

---

**그림 1.8**

예제 1.10의 에너지 소비 데이터의 히스토그램  

<img src="/assets/img/books/prob-stat-eng/1/image_10.png" alt="image" width="720px"> 

---

히스토그램으로부터,  

9보다 작은 관측값의 비율은 다음과 같다.  

$$
9보다 작은 관측값의 비율
\approx 0.01 + 0.01 + 0.12 + 0.23 = 0.37
\quad (\text{정확한 값} = \frac{34}{90} = 0.378)
$$

9 이상 11 미만 구간의 상대도수는 약 0.27이다.  
따라서 그 절반 정도, 즉 0.135가 9와 10 사이에 해당한다고 추정할 수 있다.  

따라서,  

$$
10보다 작은 관측값의 비율
\approx 0.37 + 0.135 = 0.505
\quad (\text{약 50%보다 약간 많음})
$$

이 비율의 정확한 값은 다음과 같다.  

$$
\frac{47}{90} = 0.522
$$
