# Insight Study List

Preparation materials for data science interviews, organized by **importance to interview success**.

This repo contains two era-specific study guides:
* [**2016 Edition**](README_2016.md) - Classical data science (Insight Fellowship era)
* [**2026 Edition**](README_2026.md) - Modern data science / AI engineering

Below is a unified, importance-ranked view across both.

---

**Priority Key:** Topics are ranked by interview importance.
* **[!!!]** - Must know. You will be asked about this.
* **[!!]** - Should know. Comes up frequently.
* **[!]** - Nice to know. Differentiates strong candidates.

---

## Tier 1: Must Know [!!!]

### SQL
The single most universal technical screen. Every role, every company.
* Window functions (`RANK()`, `LEAD()`, `LAG()`, `ROW_NUMBER()`)
* CTEs / `WITH` clauses
* Joins (inner, left, full outer) and aggregations (`GROUP BY`, `HAVING`)
* Query readability: clean formatting, capitalized keywords

### Statistics & Experimentation
The "science" in data science. Weak stats = instant rejection.
* A/B testing end-to-end: hypothesis, sample size, power, T-test, interpretation
* Metric guardrailing: improving one metric without degrading others
* p-values, confidence intervals, Type I/II errors
* When to use mean vs. median, normal vs. non-normal assumptions

### Classical ML (Tabular Data)
Still the bread and butter for most business problems.
* XGBoost / LightGBM - the default for structured data
* Linear & logistic regression (know the math, know the assumptions)
* Model selection: cross-validation, bias-variance tradeoff
* Feature engineering and selection

### Python Fundamentals
You will live-code in Python. Fluency is assumed.
* Data structures: lists, dicts, sets, and their complexities
* Pandas / Polars for data manipulation
* Writing clean, modular code (not notebook spaghetti)

---

## Tier 2: Should Know [!!]

### Generative AI & LLMs
The new baseline expectation for 2026 candidates.
* Transformer intuition: what they are, why they fail, how context windows work
* RAG: connecting a vector DB to an LLM (the "Hello World" of applied AI)
* Prompt engineering: system prompts, few-shot, structured outputs
* Evaluation: golden datasets, semantic similarity, exact match
* Fine-tuning vs. prompting: when to use each, cost tradeoffs

### Software Engineering
The biggest gap for PhD candidates. Companies care about this more than you think.
* Production code: modules > notebooks, type hints, pytest
* APIs: FastAPI basics - serve a model behind an endpoint
* Docker: package your work so it runs anywhere
* Git: branches, PRs, not just add/commit/push

### Algorithms & Data Structures
Less weight than 2016, but still appears in technical screens.
* Big-O notation and common complexities
* Sorting: know merge sort and quicksort cold
* Search: binary search, BFS, DFS
* Dynamic programming basics (memoization, common patterns)

### Probability & Bayes
Comes up in product-sense and ML questions alike.
* Bayes' theorem and conditional probability
* Common puzzles: disease testing, Monty Hall
* Probability distributions and when to apply them

---

## Tier 3: Nice to Know [!]

### MLOps & Deployment
Differentiates "I can build a model" from "I can ship a model."
* Experiment tracking: MLflow or Weights & Biases
* Pipeline orchestration: Airflow or Dagster
* Cloud ML: SageMaker / Vertex AI / Bedrock (pick one, know concepts)

### System Design for ML
Increasingly common in senior-level loops.
* "Design a recommendation system"
* "Design a RAG pipeline"
* Cost / latency / quality tradeoffs

### Product & Business Analytics
Critical for product data scientist roles.
* Funnel analysis, cohort analysis, churn, LTV
* Correlation vs. causation in business context
* "What conclusions can you draw from this graph?"

### Modern Data Formats & Tools
Shows you're current.
* Parquet vs. CSV, JSONL for LLM datasets
* DuckDB for local analytics
* LangChain / LlamaIndex for LLM orchestration

### Responsible AI & Ethics
Asked more often than candidates expect.
* Hallucination mitigation and guardrails
* Bias detection in models and datasets
* When *not* to use ML

---

## Interview Mindset

> In 2015, the question was *"Can you derive the math?"*
>
> In 2026, the question is *"Can you build a system that uses this math without costing us $1M in API credits?"*

**General advice:**
* Know your resume and projects inside-out. Speak with conviction.
* Research each company: what data problems do they solve? What would you build there?
* Practice communicating technical work to non-technical audiences.
* Write clean SQL and clean Python during screens - style matters.
