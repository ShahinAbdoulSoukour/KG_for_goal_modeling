[![python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

## Description

The algorithm incrementally constructs sets of triples (i-triples) and evaluates their entailment with the goal using a natural language inference (NLI) model.
In this approach, **G2T is not used**.

### Key concepts
i-Triple (T)

A set of i triples, composed of:
- One anchor triple
- Up to i-1 neighbor triples (context)

### Key features
Each anchor triple is evaluated against the goal using a NLI model:
- If entailing, it is stored and marked as retained.
- If non-entailing, context expansion begins.

#### Context Expansion for Non-Entailing Triples
- The non-entailing anchor forms a 1-triple T.
- Neighbor triples are ranked by similarity. A beam width of top candidates is selected.
- For each neighbor triple n:
  - If T ∪ {n} is already in ENT and entailing → skip.
  - If n itself is entailing → exclude and expand beam.
  - Sentiment-prefixed if needed.
  - Evaluate T ∪ {n}:
    - If entailing, check if any subset that includes n is also entailing → retain the smallest entailing subset.
    - If none are entailing but score improves, continue expanding the triple set with more neighbors (up to a depth limit).


### General Step
Repeats the same logic for existing i-triples (i > 1):
- Add a new neighbor n.
- Skip if any subset including n is already entailing.
- Evaluate T ∪ {n} and its subsets.
- Retain the smallest subset if entailing.
- If score improves, continue expansion.


## Installing the dependencies

Inside a dedicated Python environment:

```shell
pip install -r requirements.txt
```

## Run the tool

```shell
uvicorn main:app
```

The tool is then accessible by opening a webpage at the URL [127.0.0.1:8000](http://127.0.0.1:8000) or [localhost:8000](http://localhost:8000)

## Increasing the speed of the inferences using HuggingFace Inference Endpoints

If you use HuggingFace Inference Endpoints, you can perform the NLI and sentiment analysis tasks on remote servers by creating a `.env` file at the root of this project and adding the following environment variables:

- `HF_TOKEN`: Your HuggingFace Inference Endpoints access token
- `API_URL_NLI`: The URL to your endpoint containing a model dedicated to NLI (in our paper, we use `ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli`)
- `API_URL_SENT`: The URL to your endpoint containing a model dedicated to sentiment analysis (in our paper, we use `cardiffnlp/twitter-roberta-base-sentiment-latest`)
