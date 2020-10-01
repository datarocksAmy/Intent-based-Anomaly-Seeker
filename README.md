# Intent-based Anomaly Seeker

## Project Overview
Find outlier/anomaly for multi-class intents using simple Doc2Vec.

## Project Purpose
This is intend for a code challenge.
A better solution would be applying a NLU Engine with more correct tagged intend and entity. ( such as [Snips NLU](https://github.com/snipsco/snips-nlu) or [Rasa NLU](https://github.com/RasaHQ/rasa)) since it is more of a natural language understanding - e.g. chatbox, voiced-base conversation.
That being said, there's definitely still a lot of space for improvement!

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
