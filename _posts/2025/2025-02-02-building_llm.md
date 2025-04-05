---
title: (Book Review) Building LLMs for Production
author: oksjjj
date: 2025-02-02 13:03:00 +0900
categories: [Book Review]
tags: [llm, chat gpt, transformer, openai, langchain, llama, claude, book, 책, review, 리뷰]
render_with_liquid: false
image:
  path: /assets/img/thumbnail/building_llm.png
  alt: (Book Review) Building LLMs for Production
  src: "https://oksjjj.github.io/thumbnail/building_llm.png"
---

## **읽게 된 계기**

나름대로 매우 조잡하게 Elastic search를 이용해서 RAG를 구축했었다  

그런데 이렇게 주먹구구 식으로 하지 말고 제대로 배우고 싶어서  
교보문고에서 LLM과 RAG를 제대로 배울 수 있는 책을 찾아 봤는데, 찾을 수가 없었다  

그래서, 어쩔 수 없이 거금을 들여서 영어로 된 책을 아마존에서 구입했다  

## **목차**

1. Introduction to Large Language Models  
2. LLM Architectures & Landscape  
3. LLMs in Practice  
4. Introduction to Prompting  
5. Retrieval-Augmented Generation  
6. Introduction to LangChain & LlamaIndex  
7. Prompting with LangChain  
8. Indexes, Retrievers, and Data Preparation  
9. Advanced RAG  
10. Agents  
11. Fine-Tuning  
12. Deployment and Optimization  

## **소감**

### **LLM이 무엇인지? 어떻게 발전해가고 있는지? 알 수 있게 해준다**

LLM은 transformer의 decoder 모듈을 이용해서 만든다는 것은 알고 있었다  

그런데 이 책을 통해서 새로 배운게 있다  

1. LLM에는 Emergent Abilities 라는 것이 있다  

   학습 시에는 전혀 의도하지 않았던 능력들이  
   대규모의 데이터를 통해 학습을 시키다 보면 나타난다는 것이다  
   단순히 다음 단어를 예측하도록 엄청나게 많은 텍스트를 통해 학습을 시키다 보니,  
   전혀 의도하지 않았던 번역 능력이 생긴다거나, 감정 분류 능력이 생긴다는 것이다  

2. LLM의 token window를 획기적으로 늘리기 위해, sparse transformer가 연구되고 있다  

   transformer는 query 벡터와 key 벡터간 연산을 위해서 토큰 길이의 제곱 만큼의 연산량이 필요하다  
   따라서 LLM이 처리할 수 있는 token window에는 한계가 생긴다는 것이다  
   그래서 희소한 query, key 벡터를 만들어서 token window를 획기적으로 키우는 기술이 연구되고 있다고 한다  

3. Cluade가 OpenAI보다 말을 조목조목 잘한다고 느꼈는데, 그 이유가 있었다

   Cluade를 만든 Anthropic 이라는 회사는 OpenAI 직원들이 나와서 만든 회사라고 한다  
   Cluade는 RLHF(Reinforcement Learning with Human Feedback) 이라는 기술로 학습된다고 한다  
   Cluade가 만들어 내는 대답을 인간이 평가함으로써, helpfulness와 harmlessness가 되도록 강화학습을 진행한다  


### **Prompt Engineering을 실습해 볼 수 있다**

langchain을 통해서 객체를 만들고, 메소드를 통해서 Few shot prompting 들을 하는 법을 배운다  


### **RAG를 실습해 볼 수 있다**

langchain, llamaindex, deeplake를 이용해서,  
텍스트 자료를 쪼개서 임베딩 한 이후에 deeplake에 저장하고,  
질문에 대해 유사한 자료를 검색한 이후에 질문하는 것을 실습한다  

### **Agent를 실습해 볼 수 있다**

LLM과 상호 작용을 통해서 점점 논리를 정교화해 가는 과정을 실습해 볼 수 있다  
이건 마치 AI에 진짜 지능이 있는 것처럼 느껴졌다  

### **양자화가 뭔지 배울 수 있다**

딥러닝 관련 공부할 때 양자화 라는 얘기를 얼핏 듣긴 했는데 정확하게 뭔지는 잘 몰랐다  
이 책의 후반 부분에 그 설명이 나온다  
파라미터 관련 메모리를 줄이기 위해서 양자화를 시킨다는 것이다  
만약에 1 에서 100,000까지 모두 다 보관하려면 100,000의 숫자를 보관할 수 있는 데이터 타입이 필요할 것이다  
그런데 100 이라는 구간 별로 나눠서 숫자를 변환시키면...  
(예를 들어서, 180 이라는 숫자는 반올림 해서 200으로 변환)  
100,000 / 100 = 1,000  
즉, 1,000 이라는 숫자를 저장할 수 있는 데이터 타입만 있으면 될 것이다  

약간의 데이터 유실은 감수하고라도 더 큰 숫자의 파라미터를 가진 모델을 학습시킬 수 있기 때문에 메리트가 있는 것으로 보인다  

### **코드를 제대로 돌리기가 매우 어렵다**

2024년 2월에 나온 책인데도  
10 ~ 20% 정도는 제대로 돌아가지 않는다  

perplexity를 붙잡고, "최신 버전에서는 이 코드를 어떻게 바꿔야 해?" 만 물어보고 있다  

확실히 기술 발전이 빠른 분야라서 그런가 보다.....