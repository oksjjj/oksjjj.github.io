---
title: Ch#5 - BUILDING LLMS FOR PRODUCTION 코드 현행화
author: oksjjj
date: 2025-04-12 15:20:00 +0900
categories: [LLM/RAG]
tags: [LLM, RAG, langchain, chat, gpt, llama, index, deeplake]
render_with_liquid: false
image:
  path: /assets/img/thumbnail/building_llms_code.png
  alt: BUILDING LLMS FOR PRODUCTION 코드 현행화
  src: "https://oksjjj.github.io/building_llms_code.png"
---

  
### Prompt Templates

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = "You are an assistant that helps users find information about movies."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "Find information about the movie {movie_title}."
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                               human_message_prompt])

response = chat.invoke(chat_prompt.format_prompt(movie_title="Inception").to_messages())

print(response.content)
```

### Summarization Chain Example

```python
```