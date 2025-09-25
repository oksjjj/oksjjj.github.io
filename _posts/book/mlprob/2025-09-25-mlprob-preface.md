---
layout: post
title: "[서문] Machine Learning: A Probability Perspective"
date: 2025-09-25 10:00:00 +0900
categories:
  - "Books"
  - "Machine Learning: A Probability Perspective"
tags: []
---


# 서문 (Preface)

---

## 소개 (Introduction)

전자적 형태로 존재하는 데이터의 양이 점점 증가함에 따라, 자동화된 데이터 분석 방법의 필요성도 계속 커지고 있다.  
머신러닝의 목표는 데이터 속에서 자동으로 패턴을 탐지할 수 있는 방법을 개발하고, 그 후 이러한 패턴을 활용하여 미래의 데이터나 관심 있는 다른 결과를 예측하는 것이다.  

따라서 머신러닝은 통계학과 데이터 마이닝 분야와 밀접하게 연관되어 있지만, 강조점과 용어 측면에서 약간의 차이가 있다.  
이 책은 해당 분야에 대해 상세하게 소개하고 있으며, 분자생물학, 텍스트 처리, 컴퓨터 비전, 로보틱스와 같은 응용 영역에서 가져온 실제 예제를 포함한다.  

---

## 대상 독자 (Target audience)

이 책은 고학년 학부생, 대학원 초년생(컴퓨터 과학, 통계학, 전기공학, 계량경제학 등 전공) 또는 이에 상응하는 수학적 배경을 갖춘 누구에게나 적합하다.  
특히, 독자는 기본적인 다변량 미적분학, 확률론, 선형대수학, 컴퓨터 프로그래밍에 이미 익숙한 것으로 가정된다. 통계학에 대한 사전 지식은 도움이 되지만 반드시 필요하지는 않다.  

---

## 확률론적 접근 (A probabilistic approach)

이 책은, '데이터를 학습할 수 있는 기계'를 만드는 가장 좋은 방법은 확률 이론의 도구를 사용하는 것이라는 관점을 채택하고 있다. 확률 이론은 수세기 동안 통계학과 공학의 기초로 사용되어 왔다. 확률 이론은 불확실성이 관련된 모든 문제에 적용할 수 있다. 머신러닝에서 불확실성은 여러 형태로 나타난다: 주어진 데이터에 대해 가장 좋은 예측(또는 결정)은 무엇인가? 주어진 데이터에 대해 가장 좋은 모델은 무엇인가? 나는 다음에 어떤 측정을 수행해야 하는가? 등이다.

확률적 추론을 모든 추론 문제에 체계적으로 적용하는 것을 베이지안 접근법이라고 부르기도 한다. 그러나 이 용어는 매우 강한 반응(긍정적이거나 부정적인) 을 일으키는 경우가 많으므로 우리는 보다 중립적인 용어인 "확률론적 접근"을 선호한다. 또한 우리는 종종 최대우도추정법(Maximum Likelihood Estimation)과 같은 기법을 사용하며, 이는 베이지안 방법은 아니지만 확률론적 패러다임에 분명히 속한다.

이 책은 여러 가지 임시방편적 기법들을 나열하기보다는, 기계학습에 대한 원칙적인 모델 기반 접근을 강조한다. 어떤 주어진 모델에 대해서, 다양한 알고리즘을 적용할 수 있는 경우가 많다. 반대로, 특정 알고리즘도 여러 가지 모델에 적용할 수 있는 경우가 많다. 모델과 알고리즘을 구분하는 이러한 모듈화는 좋은 교육 방법이자 좋은 엔지니어링이다.

우리는 종종 우리의 모델을 간결하고 직관적인 방식으로 명시하기 위해 그래프 모델 언어를 사용할 것이다. 이해를 돕는 것 외에도, 앞으로 보게 되겠지만, 그래프 구조는 효율적인 알고리즘을 개발하는 데 도움이 된다. 그러나 이 책은 주로 그래프 모델에 관한 책이 아니다. 이것은 확률론적 모델링 전반에 관한 책이다.

---

## 실용적 접근 (A practical approach)

이 책에서 설명된 방법의 거의 모든 것은 PMTK라는 MATLAB 소프트웨어 패키지로 구현되었으며, 이는 확률적 모델링 툴킷(Probabilistic Modeling Toolkit)을 의미한다. 이 툴킷은 pmtk3.google-code.com에서 자유롭게 다운로드할 수 있다 (여기서 숫자 3은 툴킷의 세 번째 버전을 나타내며, 이 책에서 사용된 버전이다). 또한, pmtksupport.googlecode.com에서 다른 사람들이 작성한 다양한 파일들도 사용할 수 있다. 이 파일들은 PMTK 웹사이트에 설명된 설정 방법대로 했다면, 자동으로 다운로드된다.

MATLAB은 수치 계산 및 데이터 시각화에 이상적으로 적합한 고급 인터랙티브 스크립트 언어이며, www.mathworks.com에서 구매할 수 있다. 일부 코드에는 별도로 구매해야 하는 통계 툴박스가 필요하다. 또한, Octave라는 무료 버전의 MATLAB이 있으며, 이는 http://www.gnu.org/software/octave/에서 제공되며, MATLAB의 대부분의 기능을 지원한다. 이 책의 일부(모두는 아님) 코드는 Octave에서도 작동한다. 자세한 내용은 PMTK 웹사이트를 참조하라.

PMTK는 이 책에서 많은 그림을 생성하는 데 사용되었으며, 이 그림들의 소스 코드는 PMTK 웹사이트에 포함되어 있어, 독자가 데이터나 알고리즘 또는 매개변수 설정을 변경한 효과를 쉽게 볼 수 있게 한다. 책에서는 naiveBayesFit 등과 같은 이름으로 파일을 지칭한다. 해당 파일을 찾으려면 두 가지 방법을 사용할 수 있다: MATLAB 내에서 ```which naiveBayesFit```을 입력하면 파일의 전체 경로가 출력된다. MATLAB을 가지고 있지 않으나 소스 코드를 보고 싶을 경우, 검색 엔진을 사용하면 pmtk3.google-code.com 웹사이트의 해당 파일을 찾을 수 있다.

PMTK를 사용하는 방법에 대한 자세한 내용은 웹사이트에서 확인할 수 있으며, 최신 내용으로 업데이트된다. 이러한 방법들의 기초 이론에 대한 자세한 내용은 이 책에서 확인할 수 있다.

---

## 감사의 말 (Acknowledgments)

이렇게 큰 책은 분명 팀의 노력이 필요한 작업이다. 특히 다음 사람들에게 감사의 뜻을 전하고 싶다.
지난 6년간 사무실에서 연구에 몰두하는 동안 집안일을 지켜준 아내 Margaret; 이 책의 많은 그림을 제작하고 PMTK 코드의 대부분을 작성한 Matt Dunham; 이전 초안의 모든 페이지에 대해 매우 세밀한 피드백을 준 Baback Moghaddam; 세밀한 피드백을 제공한 Chris Williams; 그림 작업을 도와준 Cody Severinski와 Wei-Lwun Lu; 이전 초안에 유용한 의견을 준 UBC 학생 여러 세대; LaTeX 스타일 파일 사용을 허락해 준 Daphne Koller, Nir Friedman, Chris Manning; 안식년 기간 일부 동안 저를 환대해 준 Stanford University, Google Research, Skyline College; 그리고 수년간 저를 재정적으로 지원해 준 다양한 캐나다 연구 지원 기관(NSERC, CRC, CIFAR)이다.

또한, 이 책의 일부에 대해 유용한 피드백을 주었거나, 그림, 코드, 연습문제, 경우에 따라 일부 텍스트까지 공유해 준 다음 사람들에게 감사드린다:
David Blei, Hannes Bretschneider, Greg Corrado, Arnaud Doucet, Mario Figueiredo, Nando de Freitas, Mark Girolami, Gabriel Goh, Tom Griffiths, Katherine Heller, Geoff Hinton, Aapo Hyvarinen, Tommi Jaakkola, Mike Jordan, Charles Kemp, Emtiyaz Khan, Bonnie Kirkpatrick, Daphne Koller, Zico Kolter, Honglak Lee, Julien Mairal, Andrew McPherson, Tom Minka, Ian Nabney, Arthur Pope, Carl Rassmussen, Ryan Rifkin, Ruslan Salakhutdinov, Mark Schmidt, Daniel Selsam, David Sontag, Erik Sudderth, Josh Tenenbaum, Kai Yu, Martin Wainwright, Yair Weiss.