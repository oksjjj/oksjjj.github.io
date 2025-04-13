---
title: Ch#6 - BUILDING LLMS FOR PRODUCTION 코드 현행화
author: oksjjj
date: 2025-04-13 05:40:00 +0900
categories: [LLM/RAG]
tags: [LLM, RAG, langchain, chat, gpt, llama, index, deeplake]
render_with_liquid: false
image:
  path: /assets/img/thumbnail/building_llms_code.png
  alt: BUILDING LLMS FOR PRODUCTION 코드 현행화
  src: "https://oksjjj.github.io/building_llms_code.png"
---

## python 버전

3.9

### PromptTemplate

```python
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = """Answer the question based on the context below.
If the question cannot be answered using the information provided,
answer with "I don't know".
Context:Quantum computing is an emerging field that leverages quantum mechanics
to solve complex problems faster than classical computers.
...
Question:{query}
Answer:"""

prompt_template = PromptTemplate(
    input_variables=["query"],
    template=template
)

chain = prompt_template | llm | StrOutputParser()

input_data = {"query":"""What is the main advantage of quantum computing over classical computing?"""}

response = chain.invoke(input_data)

print("Question:", input_data["query"])
print("Answer:", response)
```

### FewShotPromptTemplate

```python
from langchain import FewShotPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

examples = [
    {"animal": "lion", "habitat": "savana"},
    {"animal": "polar bear", "habitat": "Artic ice"},
    {"animal": "elephant", "habitat": "African grasslands"}
]

example_template = """
Animal:{animal},
Habitat:{habitat}
"""

example_prompt = PromptTemplate(
    input_variables=["animal", "habitat"],
    template=example_template
)

dynamic_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Identify the habitat of the given animal",
    suffix="Animal:{input}\nHabitat:",
    input_variables=["input"],
    example_separator="\n\n",
)

chain = dynamic_prompt | llm | StrOutputParser()

input_data = {"input": "tiger"}
response = chain.invoke(input_data)

print(response)
```

### save template in JSON

save

```python
prompt_template.save("awesome_prompt.json")
```

load

```python
from langchain.prompts import load_prompt
load_prompt = load_prompt("awesome_prompt.json")
```

### Dynamic Prompt

```python
from langchain import FewShotPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

examples = [
    {
        "query": "How do I become a better programmer?",
        "answer": "Try talking to a rubber duck; it works wonders."
    }, {
        "query": "Why is the sky blue?",
        "answer": "It's nature's way of preventing eye strain."
    }
]

example_template = """
User:{query}
AI:{answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

prefix = """The following are excerpts from conversations with an AI assistant.
The assistant is typically sarcastic and witty,
producing creative and funny responses to users' questions.
Here are some examples:
"""

suffix = """
User:{query}
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

input_data = {"query": "How can I learn quantum computing?"}
response = chain.invoke(input_data)

print(response)
```

### LengthBasedExampleSelector

```python
examples = [
    {
        "query": "How do you feel today?",
        "answer": "As an AI, I don't have feelings, but I've got jokes!"
    }, {
        "query": "What is the speed of light?",
        "answer": """ Fast enough to make a round trip around Earth 7.5 times in one second!"""
    }, {
        "query": "What is a quantum computer?",
        "answer": """ A magical box that harnesses the power of subatomic particles to solve complex problems."""
    }, {
        "query": "Who invented the telephone?",
        "answer": "Alexander Graham Bell, the original 'ringmaster'."
    }, {
        "query": "What programming language is best for AI development?",
        "answer": "Python, because it's the only snake that won't bite."
    }, {
        "query": "What is the capital of France?",
        "answer": "Paris, the city of love and baguettes."
    }, {
        "query": "What is photosynthesis?",
        "answer": """ A plant's way of saying 'I'll turn this sunlight into food. You're welcome, Earth.'"""
    }, {
        "query": "What is the tallest mountain on Earth?",
        "answer": "Mount Everest, Earth's most impressive bump."
    }, {
        "query": "What is the most abundant element in the universe?",
        "answer": "Hydrogen, the basic building block of cosmic smoothies."
    }, {
        "query": "What is the largest mammal on Earth?",
        "answer": """ The blue whale, the original heavyweight champion of the world."""
    }, {
        "query": "What is the fastest land animal?",
        "answer": "The cheetah, the ultimate sprinter of the animal kingdom."
    }, {
        "query": "What is the square root of 144?",
        "answer": "12, the number of eggs you need for a really big omelette."
    }, {
        "query": "What is the average temperature on Mars?",
        "answer": """ Cold enough to make a Martian wish for a sweater and a hot cocoa."""
    }
]
```

```python
from langchain.prompts.example_selector import LengthBasedExampleSelector

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=100
)
```

```python
dynamic_prompt_template = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n"
)
```

```python
from langchain import FewShotPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts.example_selector import LengthBasedExampleSelector

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

chain = dynamic_prompt_template | llm | StrOutputParser()

input_data = {"query": "Who invented the telephone?"}
response = chain.invoke(input_data)

print(response)
```

### Alternating Human/AI Messages

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = "You are a helpful assistant that translate english to pirate."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
example_human = HumanMessagePromptTemplate.from_template("Hi")
example_ai = AIMessagePromptTemplate.from_template("Argh me mateys")
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                example_human,
                                                example_ai,
                                                human_message_prompt])

chain = chat_prompt | llm | StrOutputParser()
chain.invoke("I love programming.")
```

### Few-shot Prompting

```python
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser

examples = [
    {
        "query": "What's the weather like?",
        "answer": "It's raining cats and dogs, better bring an umbrella!"
    }, {
        "query": "How old are you?",
        "answer": "Age is just a number, but I'm timeless."
    }
]

example_templates = """
User:{query},
AI:{answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_templates
)

prefix = """The following are excerpts from conversations with an AI assistant.
The assistant is known for its humor and wit,
providing entertaining and amusing response to users' questions.
Here are some exaples:
"""

suffix = """
User:{query}
AI:"""

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)
```

```python
chain = few_shot_prompt_template | llm | StrOutputParser()
chain.invoke("What's the secret to happiness?")
```

### Example Selectors

```python
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain import FewShotPromptTemplate, PromptTemplate
```

```python
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
    {"word": "energetic", "antonym": "lethargic"},
    {"word": "sunny", "antonym": "gloomy"},
    {"word": "windy", "antonym": "calm"},
]

example_template = """
Word:{word}
Antonym:{antonym}
"""

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_template
)
```

```python
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=25,
)
```

```python
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Word:{input}\nAnotonym:",
    input_variables=["input"],
    example_separator="\n\n"
)
```

### SemanticSimilarityExampleSelector

```python
pip install "deeplake[enterprise]<4.0.0"
```

```python

```