# Intent-based Anomaly Seeker

## Project Overview
Find outlier/anomaly for multi-class intents using Snips NLU.

## Project Purpose
This is intend for a code challenge! Still lots of space to improve.
Since it is more of a natural language understanding - e.g. chatbot, voiced-base conversation... problem, I decided to go with Snips NLU.

## Data Summary
List of lists within a json file in the form of - `[ Query , Class ]`
- Total Number of Classes : 150
- Total Number of Queries per Class : 150

For Example, data would look like this:
<br>

```python
[
  ["what expression would i use to say i love you if i were an italian", "translate"], 
  ["if i were mongolian, how would i say that i am a tourist", "translate"],
  ... 
]
```

## Process Overview
Started out going down the path as a pure NLP approach - remove punctuations, tokenization, lemmization, stemming, then extract weighted features through TF-IDF. <br>
I then throw these featuers into Doc2Vec and thought I can get a reasonable classification. <br>
However, it fails to work for the need of identifying "intents" behind natural languages and goes heavily toward the frequency of a word being used.
This will definitely help with identifying keywords in a broader perspective, but not so much to a broader spectrum like the intention behind it.
<br>
This is the moment I dicovered Snips NLU ( and a similar package called [Rasa NLU](https://rasa.com/).
Open sources specialized for contexual and natural language understanding!


## Requirements
Install require packages:
```
pip install -r requirements.txt
```

<br>

Install SpaCy `en_core_web_lg` corpus:
```
python -m spacy download en_core_web_lg
```
*_Windows user might need to go through command line w/ Admin to install under the virtual environment of this project_

## Execution
In the terminal, type in:
```bash
python p2.py data.json
```


