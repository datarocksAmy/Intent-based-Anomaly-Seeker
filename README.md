# Intent-based Anomaly Seeker
***
## Project Overview
Find outlier/anomaly for multi-class intents using simple Doc2Vec.

## Data Summary
List of lists within a json file in the form of - `[ Query , Class ]`
- Total Number of Classes : 150
- Total Number of Queries per Class : 150

For Example, data.json would look like this:
<br>

```python
[
  ["what expression would i use to say i love you if i were an italian", "translate"], 
  ["if i were mongolian, how would i say that i am a tourist", "translate"],
  ... 
]
```
