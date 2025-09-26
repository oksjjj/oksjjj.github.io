---
layout: post
title: "[텍스트 마이닝] 5. Representing Texts with Vectors 1"
date: 2025-09-26 20:00:00 +0900
categories:
  - "대학원 수업"
  - "텍스트 마이닝"
tags: []
---

> 출처: 텍스트 마이닝 – 강성구 교수님, 고려대학교 (2025)

## p7. 우리의 첫 번째 계획: 텍스트를 벡터로 표현하기

- **왜 벡터인가?**  
  - 예를 들어, 우리 상점의 각 상품은 텍스트 설명을 가진다.  
    (예: “Nike Air Force”, “Harry Potter book”)  
  - 사용자는 자연어 질의를 이용해 검색한다.  
    (예: “농구에 가장 좋은 신발”, “Harry Potter 첫 번째 책”)  
  - 항목을 분류하거나 검색하기 위해, 시스템은 텍스트를 수학적으로 비교할 수 있는 방법이 필요하다.  

- 우리는 텍스트를 **고차원 공간(high-dimensional space)** 의 벡터로 표현한다.  
  이렇게 하면 그 의미(semantics)를 포착하고 비교할 수 있다.  

<img src="/assets/img/textmining/5/image_1.png" alt="image" width="600px">  

---

## p8. 우리의 첫 번째 계획: 텍스트를 벡터로 표현하기

- 우리는 텍스트를 **고차원 공간(high-dimensional space)** 의 벡터로 표현한다.  
  이렇게 하면 그 의미(semantics)를 포착하고 비교할 수 있다.  

- 예를 들어, 어떤 상품이 질의(query)와 더 관련성이 높은 경우는  
  벡터 공간에서 서로 더 **“유사(similar)”** 할 때이다.  
  - “similar”의 의미에 대해서는 나중에 논의할 것이다.  

- 축(axes)은 무엇인가?  

- **어휘집(vocabulary)의 각 단어를 하나의 차원으로 표현한다면 어떨까?**  

<img src="/assets/img/textmining/5/image_2.png" alt="image" width="600px">  

---

# 텍스트를 벡터로 표현하기: 희소 표현 (Sparse representation)

- 단순화를 위해, 우리는 제품 설명(예: 제목, 특성)을 문서(documents)라고 부른다.  

---

## p10. 단어들의 집합 (Bag-of-words, BoW)

- **BoW 모델**은 문서를 **단어 개수(빈도, word counts / frequencies)**의 벡터로 표현한다.  
  - 어휘집(vocabulary)은 말뭉치(corpus) 전체에서의 **고유한 단어들의 집합**으로 미리 정의된다.  
  - BoW는 **단어의 순서와 문맥(context)을 무시**하고, 각 단어가 몇 번 등장하는지만 집중한다. 

<img src="/assets/img/textmining/5/image_3.png" alt="image" width="600px">  

---

## p11. 단어들의 집합 (Bag-of-words, BoW)

*우리는 ‘term’, ‘word’, ‘token’을 서로 바꿔 사용할 수 있는 용어로 사용한다.*

---

✔ 용어-문서 빈도 행렬 (Term-document count matrix)

- **각 행(row)** 은 하나의 **용어(term)** 에 대응한다.  
- **각 열(column)** 은 하나의 **문서(document)** 에 대응한다.  
- 행렬의 값은 단순히 해당 문서에서 **용어(term)가 등장한 횟수** 를 의미한다.  

<img src="/assets/img/textmining/5/image_4.png" alt="image" width="600px">  

---

## p12. 단어들의 집합 (Bag-of-words, BoW)

✔ 용어-문서 빈도 행렬 (Term-document count matrix)

- **각 행(row)** 은 하나의 **용어(term)** 에 대응한다.  
- **각 열(column)** 은 하나의 **문서(document)** 에 대응한다.  
- 행렬의 값은 단순히 해당 문서에서 **용어(term)가 등장한 횟수** 를 의미한다.  

---

<img src="/assets/img/textmining/5/image_5.png" alt="image" width="600px">  

---

- 문서는 **어휘(vocabulary) 크기에 기반한 카운트 벡터(count vectors)** 로 변환된다.  
- 실제로는 어휘 크기가 매우 크기 때문에, 이 벡터들 안의 많은 항목들이 **0**이 된다.  
  → 따라서 **희소 벡터(sparse vectors)** 가 된다.  

---

## p13. 단어들의 집합 (Bag-of-words, BoW): 문서 벡터 시각화

- **문서 벡터 시각화 (Visualizing document vectors)**  
  - 어휘(vocabulary): {nike, love}  

---

<img src="/assets/img/textmining/5/image_6.png" alt="image" width="720px">  

---

- 문서는 어휘 크기에 기반한 **카운트 벡터(count vectors)** 가 된다.  
- 이러한 벡터들은 문서 내 **용어 분포(term distributions)** 를 반영한다.  

---

## p14. 단어들의 집합 (Bag-of-words, BoW)

- **용어-문서 카운트 행렬 (Term-document count matrix)**  
  - 각 행(row)은 **용어(term)** 에 대응하고, 각 열(column)은 **문서(document)** 에 대응한다.  
  - 값(value)은 단순히 해당 문서에서 **용어(term)가 등장한 횟수** 를 의미한다.  

---

<img src="/assets/img/textmining/5/image_7.png" alt="image" width="600px">  

---

- 단어 또한 벡터로 표현될 수 있다    
- 'love' → 음악 앨범이나 로맨틱한 콘텐츠와 관련된 문서에서 자주 등장한다.  
- 'shoes' → 스니커즈 제품 설명에서 자주 등장한다.  

---

## p15. 단어에 대한 더 일반적인 선택

- **단어-단어 동시발생 행렬 (Word-word co-occurrence matrix)**  
  - ‘용어-맥락 행렬(term-context matrix)’이라고도 불린다.  
  - 크기는 $V \times V$이며, 여기서 $V$는 어휘(vocabulary)의 크기이다.  
  - 각 항목(entry)은 한 단어(행, row)가 다른 단어(열, column)와 **맥락 창(context window)** 안에서 얼마나 자주 함께 나타나는지를 센다.  

---

- *맥락 창(context window)은 몇 개의 이웃 단어들을 고려할지를 정의한다.* 
<img src="/assets/img/textmining/5/image_8.png" alt="image" width="520px">   

---

<img src="/assets/img/textmining/5/image_9.png" alt="image" width="720px">   

---

## p16. 단어에 대한 더 일반적인 선택

- **단어-단어 동시발생 행렬 (Word-word co-occurrence matrix)**  
  - ‘용어-맥락 행렬(term-context matrix)’이라고도 불린다.  
  - 크기는 $V \times V$이며, 여기서 $V$는 어휘(vocabulary)의 크기이다.  
  - 각 항목(entry)은 한 단어(행, row)가 다른 단어(열, column)와 **맥락 창(context window)** 안에서 얼마나 자주 함께 나타나는지를 센다.  

---

<img src="/assets/img/textmining/5/image_8.png" alt="image" width="520px">   

---

<img src="/assets/img/textmining/5/image_10.png" alt="image" width="720px">  

- 두 단어는, 만약 이들의 **맥락 벡터(context vectors)** (행렬의 행)이 유사하다면, 서로 유사한 것으로 간주된다.  

---

## p17. Bag-of-words (BoW): 한계점

- **문제:** 단어 빈도(word frequency)가 항상 좋은 표현(representation)인 것은 아니다.  

  - **빈도는 분명히 유용하다.** 두 문서가 유사한 단어 빈도 분포를 가진다면, 의미적으로 유사할 가능성이 크다.  
  - 그러나 **매우 흔한 단어들** (예: the, it)은 문서의 실제 내용을 거의 알려주지 않는다.  
  - 또한, 유사한 주제(예: 앨범)를 가진 **말뭉치(corpus)** 에서는 **특정 주제와 관련된 용어들** (예: song, singer)이 모든 문서에 걸쳐 자주 등장할 수 있다.  

---

<img src="/assets/img/textmining/5/image_11.png" alt="image" width="450px">  
<img src="/assets/img/textmining/5/image_12.png" alt="image" width="450px">  

---

## p18. Bag-of-words (BoW): 한계점

- **문제:** 단어 빈도(word frequency)가 항상 좋은 표현(representation)인 것은 아니다.  

  - **빈도는 분명히 유용하다.** 두 문서가 유사한 단어 빈도 분포를 가진다면, 의미적으로 유사할 가능성이 크다.  
  - 그러나 **매우 흔한 단어들** (예: the, it)은 문서의 실제 내용에 대한 정보를 거의 제공하지 않는다.  
  - 또한, 유사한 주제(예: 앨범)를 가진 **말뭉치(corpus)** 에서는 **특정 주제와 관련된 용어들** (예: song, singer)이 모든 문서에서 자주 등장할 수 있다.  

---

<img src="/assets/img/textmining/5/image_11.png" alt="image" width="450px">  
<img src="/assets/img/textmining/5/image_12.png" alt="image" width="450px">  

---

- 이것은 **역설(paradox)** 을 만든다!  
  - **빈도는 유용하다.** 그러나 **지나치게 자주 등장하는 단어들은 오해를 불러일으킬 수 있다.**  
  - 그렇다면 우리는 **이 두 상충되는 제약 조건을 어떻게 균형(balance)** 잡을 수 있을까?  
