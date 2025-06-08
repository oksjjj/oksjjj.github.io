---
title: Ch#6 - BUILDING LLMS FOR PRODUCTION 코드 현행화
author: oksjjj
date: 2025-04-13 05:40:00 +0900
categories: [LLM/RAG]
tags: [LLM, RAG, langchain, chat, gpt, llama, index, deeplake]
render_with_liquid: false
---

### python 버전

3.9

```python
from dotenv import load_dotenv
import os
load_dotenv('../env')

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['ACTIVELOOP_TOKEN'] = os.getenv('ACTIVELOOP_TOKEN')
```

### PromptTemplate

```python
from langchain_core.prompts import PromptTemplate
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
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
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

주피터 노트북 재시작 후 모두 실행

```python
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import DeepLake
from langchain.embeddings import OpenAIEmbeddings
from langchain import FewShotPromptTemplate, PromptTemplate

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template ="Input:{input}\nOutput:{output}",
)

examples = [
    {"input": "0°C", "output": "32°F"},
    {"input": "10°C", "output": "50°F"},
    {"input": "20°C", "output": "68°F"},
    {"input": "30°C", "output": "86°F"},
    {"input": "40°C", "output": "104°F"},
]

my_activeloop_org_id = "oksjjj"
my_activeloop_dataset_name = "langchain_course_fewshot_selector"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

db = DeepLake(dataset_path=dataset_path)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, embeddings, db, k=1
)

similar_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Convert the temperature from Celsius to Fahrenheit",
    suffix ="Input:{temperature}\nOutput:",
    input_variables =["temperature"],
)

print(similar_prompt.format(temperature ="10°C"))
print(similar_prompt.format(temperature ="30°C"))

similar_prompt.example_selector.add_example({"input": "50°C", "output": "122°F"})

print(similar_prompt.format(temperature ="40°C"))
```

### PydanticOutputParser

validator 는 field_validator로 변경

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from typing import List

class Suggestions(BaseModel):
    words:List[str] = Field(description="""list of substitute words based on context""")

    @field_validator('words')
    def not_start_with_number(cls, field):
        for item in field:
            if item[0].isnumeric():
                raise ValueError("The word can not start with numbers!")
        return field
    
parser = PydanticOutputParser(pydantic_object=Suggestions)
```

```python
from langchain_core.prompts import PromptTemplate

template = """
Offer a list of suggestions to substitute the specified target_word based the presented context.
{format_instructions}
target_word={target_word}
context={context}
"""

target_word = "behaviour"
context = """The behaviour of the students in the classroom was disruptive 
and made it difficult for the teacher to conduct the lesson"""

prompt_template = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model_name = "gpt-3.5-turbo"
temperature = 0
model = ChatOpenAI(model_name=model_name, temperature=temperature)

chain =  prompt_template | model | StrOutputParser()

output = chain.invoke({"target_word": target_word, "context": context})

parser.parse(output)
```

### Multiple Outputs Example

```python
template = """
Offer a list of suggestions to substitute the specified target_word
based on the presented context and the reasoning for each word.
{format_instructions}
target_word={target_word}
context={context}
"""
```

```python
class Suggestions(BaseModel):
    words:List[str]=Field(description="""list of substitute words based on context""")
    reasons:List[str]=Field(description="""list of each substitute word's reasons why this each word fits the context""")

    @field_validator('words')
    def not_start_with_number(cls, field):
        for item in field:
            if item[0].isnumeric():
                raise ValueError("The word can not start with numbers!")
        return field
    
    @field_validator('reasons')
    def end_with_dot(cls, field):
        for idx, item in enumerate(field):
            if item[-1] != ".":
                field[idx] += "."
        return field
```

```python
parser = PydanticOutputParser(pydantic_object=Suggestions)

prompt_template = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt_template | model | StrOutputParser()

output = chain.invoke({"target_word":target_word, "context": context})
parser.parse(output)
```

### CommaSeparatedOutputParser

```python
from langchain.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()
```

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

template = """
Offser a list of suggestions to substitute the word '{target_word}'
based on the following text:
{context}.
{format_instructions}
"""

prompt_template = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt_template | model | StrOutputParser()

output = chain.invoke({"target_word": target_word, "context": context})

parser.parse(output)
```

### OutputFixingParser

"reasons" 가 있어야 하는데, "reasoning" 이 있어서 에러 발생

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class Suggestions(BaseModel):
    words: List[str] = Field(description="""list of substitute words based on context""")
    reasons: List[str] = Field(description="""list of each substitute word's reasons why this each word fits the context""")

parser = PydanticOutputParser(pydantic_object=Suggestions)

missformatted_output = '{"words":["conduct","manner"],' \
'"reasoning":["refers to the way someone acts in a particular situation.",' \
'"refers to the way someone behaves in a particular situation."]}'

parser.parse(missformatted_output)
```

```python
from langchain.output_parsers import OutputFixingParser

outputfixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
outputfixing_parser.parse(missformatted_output)
```

### RetryOutputParser

```python
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class Suggestions(BaseModel):
    words: List[str] = Field(description="""list of substitute words based on the context""")
    reasons: List[str] = Field(description="""list of each substitute word's reasons why this each word fits the context""")

parser = PydanticOutputParser(pydantic_object=Suggestions)

template = """
Offer a list of suggestions to subtitute the specified target_word
based on the presented context and the reasoning for each word.
{format_instructions}
target_word={target_word}
context={context}
"""

prompt = PromptTemplate(
    template=template,
    input_variable=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format_prompt(target_word="behaviour",
              context="""The behaviour of the students in the classroom was disruptive
                         and made it difficult for the teacher to conduct the lesson.""")
```

```python
from langchain.output_parsers import RetryWithErrorOutputParser

missformatted_output = '{"words":["conduct","manner"]}'

retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=model)

retry_parser.parse_with_prompt(missformatted_output, model_input)
```

### Improving News Articles Summarizer

```python
import requests
from newspaper import Article

headers = {
    'User-Agent': """Mozilla/5.0(Windows NT 10.0;Win64;x64) AppleWebKit/537.76(KHTML,like Gecko) Chrome/89.0.4389.82 Safari/537.36"""
}

article_url = """https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-super-computer-will-set-records/"""

session = requests.Session()

try:
    response = session.get(article_url, headers=headers, timeout=10)

    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()

        print(f"Title: {article.title}")
        print(f"Text: {article.text}")

    else:
        print(f"Failed to fetch an article at {article_url}")

except Exception as e:
    print(f"Error occurred while fetching article at {article_url}: {e}")
```

```python
from langchain.schema import HumanMessage

template = """
As an advanced AI, you've been tasked to summarize online articles into bulleted points.
Here are a few examples of how you've done this in the past:

Example 1:
Original Article: 'The Effects of Climate Change
Summary:
-Climate change is causing a rise in global temperatures.
-This leads to melting ice caps and rising sea levels.
-Resulting in more frequent and severe weather conditions.

Example 2:
Original Article: 'The Evolution of Artificial Intelligence
Summary:
-Artificial Intelligence (AI) has developed significantly over the past decade.
-AI is now used in multiple fields such as healthcare, finance, and transportation.
-The future of AI is promising but requires careful regulation.

Now, here's the article you need to summarize:

==================
Title:{article_title}

{article_text}
==================

Please provide a summarized version of the article in a bulleted list format.
"""

prompt = template.format(article_title=article.title, article_text=article.text)

messages = [HumanMessage(content=prompt)]
```

```python
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.0)

summary = chat.invoke(messages)

print(summary.content)
```

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import field_validator
from pydantic import BaseModel, Field
from typing import List

class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(description="Bulleted list summary of the article")

    @field_validator('summary')
    def has_three_or_more_lines(cls, list_of_lines):
        if len(list_of_lines) < 3:
            raise ValueError("Generated summary has less than three bullet points!")
        return list_of_lines
    
parser = PydanticOutputParser(pydantic_object=ArticleSummary)
```

```python
from langchain_core.prompts import PromptTemplate

template = """
You are a very good assistant that summarizes online articles.

Here's the article you want to summarize.

============================
Title: {article_title}

{article_text}
============================

{format_instructions}
"""

prompt_template = PromptTemplate(
    template=template,
    input_variables=["article_title", "article_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

chain = prompt_template | model | StrOutputParser()

output = chain.invoke({"article_title": article.title, "article_text": article.text})

parsed_output = parser.parse(output)
print(parsed_output)
```

### Creating Knowledge Graphs from Textual Data

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs.networkx_graph import KG_TRIPLE_DELIMITER

_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE = (
    "You are a networked intelligence helping a human track knowledge triples"
    "about all relevant people, things, concepts, etc. and integrating"
    "them with your knowledge stored within your weights"
    "as well as that stored in a knowledge graph."
    "Extract all of the knowledge triples from the text."
    "A knowledge triple is a clause that contains a subject, a predicate,"
    "and an object. The subject is the entity being described,"
    "the predicate is the property of the subject that is being"
    "described, and the object is the value of the property.\n\n"
    "EXAMPLE\n"
    """It's a state in the US. It's also the number 1 producer of gold in the US.\n\n"""
    f"Output:(Nevada,is a,state){KG_TRIPLE_DELIMITER}(Nevada,is in,US)"
    f"{KG_TRIPLE_DELIMITER}(Nevada,is the number 1 producer of,gold)\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "I'm going to the store.\n\n"
    "Output: NONE\n"
    "EXAMPLE\n"
    """Oh huh. I know Descartes likes to drive antique scooters and play the mandolin.\n"""
    f"""Output: (Descartes,likes to drive,antique scooters){KG_TRIPLE_DELIMITER}(Descartes,plays,mandolin)\n"""
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "{text}"
    "Output:"
)

KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE,
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)

chain = KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT | llm | StrOutputParser()

text = """The city of Paris is the capital and most populous city of France.
The Eiffel Tower is a famous landmark in Paris."""

triples = chain.invoke(text)

print(triples)
```

```python
def parse_triples(response, delimiter=KG_TRIPLE_DELIMITER):
    if not response:
        return []
    return response.split(delimiter)

triples_list = parse_triples(triples)

print(triples_list)
```

```bash
pip install pyvis
```

```python
from pyvis.network import Network
import networkx as nx

# Create a NetworkX graph from the extracted relation triplets
def create_graph_from_triplets(triplets):
    G = nx.DiGraph()
    for triplet in triplets:
        subject, predicate, obj = triplet.strip().split(',')
        G.add_edge(subject.strip(), obj.strip(), label=predicate.strip())
    return G

# Convert the NetworkX graph to a PyVis network
def nx_to_pyvis(networkx_graph):
    pyvis_graph = Network(notebook=True)
    for node in networkx_graph.nodes():
        pyvis_graph.add_node(node)
    for edge in networkx_graph.edges(data=True):
        pyvis_graph.add_edge(edge[0],edge[1],label=edge[2]['label'])
    return pyvis_graph

triplets = [t.strip() for t in triples_list if t.strip()]
graph = create_graph_from_triplets(triplets)
pyvis_network = nx_to_pyvis(graph)

# Customize the appearance of the graph
pyvis_network.toggle_hide_edges_on_drag(True)
pyvis_network.toggle_physics(False)
pyvis_network.set_edge_smooth('discrete')

# Show the interactive knowledge graph visualization
pyvis_network.show('knowledge_graph.html')
```