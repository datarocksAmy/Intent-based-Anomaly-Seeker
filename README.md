# Intent-based Anomaly Seeker

## Project Overview
Find outlier/anomaly for multi-class intents using Snips NLU.

## Project Purpose
This is intend for a code challenge! Still lots of space to improve. <br>
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


## Wiki
Documentation on overview, approaches, how Snips NLU fits into tackling this challenge. <br>
See [here](https://github.com/datarocksAmy/Intent-based-Anomaly-Seeker/wiki) for more details.
