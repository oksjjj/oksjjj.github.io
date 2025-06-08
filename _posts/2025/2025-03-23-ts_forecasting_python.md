---
title: (Book Review) Time Series Forecasting in Python
author: oksjjj
date: 2025-03-23 19:00:00 +0900
categories: [Book Review]
tags: [시계열 분석, ARIMA, LSTM, Python, book, 책, review, 리뷰]
render_with_liquid: false
---

## **읽게 된 계기**

시계열 분석 관련해서 어떤 온라인 수업을 듣게 되었는데,  
아무런 기초 지식이 없는 나에게, ARIMA, p, d, q 등의 용어를 쏟아내는 강의에 좌절하게 되었다  
그래서 어떤 책을 읽어야 이런 내용을 쉽게 배울 수 있는지 열심히 찾아보다가,  
"파이썬 시계열 예측 분석" 이라는 책을 찾게 되었다.  
그런데, 이 책은 번역본이었다.  
원서가 있을 땐 원서를 사자라는 내 나름의 원칙 하에 아마존 kindle 버전으로 원서를 구입하기 되었다.  
2배의 돈을 주고... ㅠㅠ  


## **목차**
 
Part 1. Time waits for no one  
1. Understanding time series forecasting  
2. A naive prediction of the future 
3. Going on a random walk 
  
Part 2. Forecasting with statistical models  
4. Modeling a moving average process
5. Modelling an autoregressive process
6. Modeling complex time series
7. Forecasting non-stationary time series
8. Accounting for seasonality
9. Adding external variables to our model
10. Forecasting multiple time series
11. Capstone: Forecasting the number of antidiabetic drug prescriptions in Australia
  
Part 3. Large-scale forecasting with deep learning  
12. Introducing deep learning for time series forcasting
13. Data windowing and creating baselines for deep learning
14. Baby steps with deep learning
15. Remembering the past with LSTM
16. Filtering a time series with CNN
17. Using predictions to make more predictions
18. Capstone: Forecasting the electric power consumptions of a household
  
Part 4. Automating forecasting at scale  
19. Automating time series forecasting with Prophet
20. Capstone: Forecasting the monthly average retail price steak in Canada
21. Going above and beyond
 
## **소감**

### **시키는 대로 따라서 타이핑 하다 보면 실력이 늘어난다**

처음엔 stationary, AR, MA 등이 무엇인지 전혀 몰라서 두려움이 있었다.  
그런데, 저자가 책을 구성할 때, 가장 쉬운 내용부터 하나씩 더해가는 식으로 잘 구성해 놓았기 때문에,  
쉬운 코드부터 하나씩 실습하다 보니,  
ARIMA, SARIMA가 무엇인지 쉽게 이해할 수 있게 되었다.   
  
  
### **시계열 예측과 관련된 모든 기술들을 일목요연하고 짜임새 있게 다루고 있다**

ARIMA, SARIMA 등 통계 기반의 기법들,  
LSTM, CNN, ARLSTM 등 딥러닝 기반의 기법들,  
그리고 Prophet과 같이 자동화된 라이브러리 등  
하나도 빠짐 없이 다루고 있어서,  
이 책 하나만 있으면 시계열 예측 관련된 거의 모든 내용을 다 배울 수 있다.  
  

### **약간은 과하다 싶을 정도로 설명이 친절하다**
  
주요 코드의 각 라인 별로, 코드의 의미가 무엇인지 설명이 달려 있고,  
stationary, AR, MA, LSTM, CNN 등 새로운 용어가 나올 때 마다 친절하게 설명해 준다.   
  
  
### **결론은 "매우 만족" 이다!!!**

간만에 매우 만족스럽게 읽고, 많은 내용을 배우게 된 책이다.