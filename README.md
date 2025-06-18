# Weighted AI Index (wAII): Automated Measurement of AI Exposure

This Python package provides tools to automatically assess the **AI exposure**, or the extent to which a job or occupation is susceptible to being impacted by artificial intelligence, either through automation of tasks or by requiring workers to adapt to new technologies. The package uses task extraction and patent similarity retrieval pipelines. It implements the measurement framework described in our [ICWSM 2025 paper](https://workshop-proceedings.icwsm.org/abstract.php?id=2025_05).

## Installation

You can install the package using either of the following:

```bash
# Install directly from GitHub
pip install git+https://github.com/EunCheolChoi0123/icwsm25-dwmv-ai-jobs.git

# Or install locally (if you've cloned the repo)
pip install .
```

## Task Extraction with `TaskExtractor`

Extract key tasks from a job description with associated importance weights.

```python
from waii import TaskExtractor

extractor = TaskExtractor(
    api_key="sk-...",      # Replace with your actual OpenAI API key
    num_tasks=5,           
    model="gpt-4o-mini",   
    temperature=0.1        
)

description = """Your Job Description Text"""
tasks = extractor.extract(description)
print(tasks)  # Output: List of (task: str, weight: float) tuples, which can be directly put into PatentRetriever Example 3.
```

## AI Exposure Retrieval with `PatentRetriever`

Retrieve semantically similar patent records and calculate weighted AI exposure (wAII) using patent embeddings.

```python
from waii import PatentRetriever

retriever = PatentRetriever()
```

### Example 1: Single task string

```python
retriever.ai_exposure("Build dashboards")
```

### Example 2: List of task strings (weights via softmax)

```python
retriever.ai_exposure([
    "Build dashboards", 
    "Conduct A/B testing", 
    "Present insights to stakeholders"
], weight="softmax")
```

### Example 3: Custom weighted tasks

```python
retriever.ai_exposure([
    ("Build dashboards", 0.3), 
    ("Conduct A/B testing", 0.5), 
    ("Present insights to stakeholders", 0.2)
])
```

**Returned fields (dict) include:**
- `past_ai_exposure` and `current_ai_exposure`: wAII scores
- `past_patent_abstracts` and `current_patent_abstracts`: Closest matched patents

## Replication Code

The `replication_code/` directory contains all scripts and notebooks used in the original ICWSM 2025 paper to:
- Generate tasks from the LinkedIn job posts
- Embed and index the patent dataset (FAISS)
- Calculate wAII for each job posting
- Run analyses (RQ1-3)

## 📄 Citation

If you use this package, please cite our paper:

> Choi, E. C. (2025). *Weighted AI Index: Measuring Task Exposure to AI via Patent Embeddings and Language Models*. In Proceedings of the 17th International AAAI Conference on Web and Social Media (ICWSM).

[View abstract and proceedings](https://workshop-proceedings.icwsm.org/abstract.php?id=2025_05)
