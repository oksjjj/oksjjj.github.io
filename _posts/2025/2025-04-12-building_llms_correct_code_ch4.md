---
title: Ch#4 - BUILDING LLMS FOR PRODUCTION 코드 현행화
author: oksjjj
date: 2025-04-12 07:30:00 +0900
categories: [LLM/RAG]
tags: [LLM, RAG, langchain, chat, gpt, llama, index, deeplake]
render_with_liquid: false
image:
  path: /assets/img/thumbnail/building_llms_code.png
  alt: BUILDING LLMS FOR PRODUCTION 코드 현행화
  src: "https://oksjjj.github.io/building_llms_code.png"
---


### python 버전

3.9
  
```bash
pip install openai==1.70.0
pip install python-dotenv
```  

```python
from dotenv import load_dotenv
import os

load_dotenv('../env')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
```
  
### Example: Story Generation

```python
import openai

prompt_system = "You are a helpful assistant whose goal is to help write stories."

prompt = """Continue the following story. Write no more than 50 words.
Once upon a time, in a world animals could speak, a courageous mouse named Benjamin decided to"""

response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)
```
  
### Example: Product Description

```python
import openai

prompt_system = """You are a helpful assistant whose goal is to help write product descriptions"""

prompt = """Write a captivationg product description for a luxurious, handcrafted, limited-edition fountail pen made from rose wood and gold.
Write no more than 50 words."""

response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)
```

### Zero-Shot Prompting

```python
import openai

prompt_system = "You are a helpful assistant whose goal is to write short peoms."

prompt = """Write a short poem about {topic}."""

response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt.format(topic="summer")}
    ]
)

print(response.choices[0].message.content)
```

### Few-Shot Prompting

```python
import openai

prompt_system = "You are a helpful assistant whose goal is to write short poems."

prompt = """Write a short poem about {topic}."""

examples = {
    "nature": """Birdsong fills the air,\nMountains high and valleys deep,\nNature's music sweet.""",
    "winter": """Snow blankets the ground,\nSilence is the only sound,\nWinter's beauty found."""
}

response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt.format(topic="nature")},
        {"role": "assistant", "content": examples["nature"]},
        {"role": "user", "content": prompt.format(topic="winter")},
        {"role": "assistant", "content": examples["winter"]},
        {"role": "user", "content": prompt.format(topic="summer")}
    ]
)

print(response.choices[0].message.content)
```

### pkg 설치
```bash
pip install langchain langchain_community langchain-openai
```

### Few-Shot Prompting Example (langchain)

```python
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Few-shot examples
examples = [
    {"color": "red", "emotion": "passion"},
    {"color": "blue", "emotion": "serenity"},
    {"color": "green", "emotion": "tranquility"}
]

# Example formatter
example_formatter_template = """
Color: {color},
Emotion: {emotion}
"""

example_prompt = PromptTemplate(
    input_variables=["color", "emotion"],
    template=example_formatter_template,
)

# Few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="""Here are some examples of colors and the emotions associated with them:\n\n""",
    suffix="""\n\nNow, given a new color, identify the emotion associated with it:\n\nColor: {input}\nEmotion:""",
    input_variables=["input"],
    example_separator="\n",
)

# Compose chain using | operator
chain = few_shot_prompt | llm | StrOutputParser()

# Run the chain
response = chain.invoke({"input": "purple"})

print("Color: purple")
print("Emotion:", response)
```

### Role Prompting

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = """
As a futuristic robot band conductor, I need you to help me come up with a song title.
What's a cool song title for a song about {theme} in the year {year}:
"""

prompt = PromptTemplate(
    input_variables=["theme", "year"],
    template=template,
)

# Input data for the prompt
input_data = {"theme": "interstella travel", "year": "3030"}

# Compose chain using | operator
chain = prompt | llm | StrOutputParser()

# Run the chain
response = chain.invoke(input_data)

print("Theme: interstellar travel")
print("Year: 3030")
print("AI-generated song title:", response)
```

### Chain Prompting

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

template_question = """What is the name of the famous scientist who developed the theory of general relativity?
Answer:"""
prompt_question = PromptTemplate(template=template_question, input_variables=[])

template_fact = """Provide a brief description of {scientist}'s theory of general relativity.
Answer:"""
prompt_fact = PromptTemplate(input_variables=["scientist"],
                             template=template_fact)

chain_question = prompt_question | llm | StrOutputParser()
response_question = chain_question.invoke({})

scientist = response_question.strip()

chain_fact = prompt_fact | llm | StrOutputParser()
response_fact = chain_fact.invoke(scientist)

print("Scientist:", scientist)
print("Fact:", response_fact)
```

### Bad Prompt Practices

```python
from langchain_core.prompts import PromptTemplate

template = "Tell me something about {topic}"
prompt = PromptTemplate(
    input_variables=["topic"],
    template=template,
)
prompt.format(topic="dogs")
```

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

template_question = """What is the name of the famous scientist who developed the theory of general relativity?
Answer:"""
prompt_question = PromptTemplate(template=template_question, input_variables=[])

template_fact = """Tell me something interesting about {scientist}.
Answer:"""
prompt_fact = PromptTemplate(input_variables=["scientist"],
                             template=template_fact)

chain_question = prompt_question | llm | StrOutputParser()
response_question = chain_question.invoke({})

scientist = response_question.strip()

chain_fact = prompt_fact | llm | StrOutputParser()
input_data = {"scientist": scientist}
response_fact = chain_fact.invoke(input_data)

print("Scientist:", scientist)
print("Fact:", response_fact)
```

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

template_question = """What are some musical genres?
Answer:"""
prompt_question = PromptTemplate(template=template_question, input_variables=[])

template_fact = """Tell me something about {genre1}, {genre2} and {genre3} without giving any specific details.
Answer:"""
prompt_fact = PromptTemplate(input_variables=["genre1", "genre2", "genre3"],
                             template=template_fact)

chain_question = prompt_question | llm | StrOutputParser()
response_question = chain_question.invoke({})

# Assign three hardcoded genres
genre1, genre2, genre3 = "jazz", "pop", "rock"

chain_fact = prompt_fact | llm | StrOutputParser()
input_data = {"genre1":genre1, "genre2":genre2, "genre3":genre3}

response_fact = chain_fact.invoke(input_data)

print("Genres:", genre1, genre2, genre3)
print("Fact:", response_fact)
```

### Tips for Effective Prompt Engineering
```python
from langchain import FewShotPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

examples = [
    {
        "query": "What's the secret to happiness?",
        "answer": """Finding balance in life and learning to enjoy the small moments."""
    }, {
        "query": "How can I become more productive?",
        "answer": """Try prioritizing tasks, and maintaining a healthy work-like balance."""
    }
]

example_template = """
User: {query}
AI: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

prefix = """The following are exerpts from conversations with an AI life coach.
The assistant provides insightful and practical advice to the user's questions.
Here are some examples:
"""

suffix = """
User: {query}
AI:"""

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

chain = few_shot_prompt_template | llm | StrOutputParser()

user_query = "What are some tips for improving communication skills?"

response = chain.invoke({"query": user_query})

print("User Query:", user_query)
print("AI Response:", response)
```