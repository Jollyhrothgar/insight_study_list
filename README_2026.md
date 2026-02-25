# 2026 Insight Study List (PhD Transition Edition)

**Target Role:** Data Scientist / Applied Scientist / AI Engineer

> **The "PhD Trap" Warning:**
>
> **Don't Study:** Deriving backpropagation by hand, obscure statistical proofs, or writing your own CUDA kernels (unless applying to OpenAI Core).
>
> **Do Study:** Git workflows, API design, Docker, and how to write code that isn't a single 5,000-line Jupyter Notebook.

---

## I. Generative AI & LLMs
*Replacing "Neural Networks / Theano"*

### Applied Architectures
* [1] Transformer Intuition: inputs, outputs, context windows, temperature. Know *why* it fails, not how to code multi-head attention from scratch
* [1] [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) (Retrieval Augmented Generation): the "Hello World" of 2026. Hook a vector DB ([Pinecone](https://www.pinecone.io/), [Milvus](https://milvus.io/)) to an LLM
* [1] Embeddings & Vector Search: understand how text becomes vectors and how similarity search works

### Prompt Engineering as Code
* [2] System prompts, few-shot prompting, and structured outputs (JSON enforcement)
* [2] [LangChain](https://www.langchain.com/) / [LlamaIndex](https://www.llamaindex.ai/) - orchestration frameworks for LLM apps
* [1] Fine-tuning vs. prompting tradeoffs: when each approach is appropriate, cost implications

### Evaluation & Safety
* [1] Eval systems: "golden datasets" of inputs/outputs, not just "vibe checks"
* [2] Exact match vs. semantic similarity metrics
* [2] Hallucination mitigation, guardrails, and responsible AI basics

---

## II. Software Engineering (The Biggest PhD Gap)
*Replacing "Web Dev / Tools"*

### Production-Ready Code
* [1] Modularization: moving from notebooks (`.ipynb`) to scripts/modules (`.py`)
* [2] Typing: Python type hints (`def func(x: list[str]) -> int:`)
* [1] Testing: [pytest](https://docs.pytest.org/) - unit tests for data pipelines are non-negotiable

### The Modern Stack
* [1] APIs: [FastAPI](https://fastapi.tiangolo.com/) (replacing Flask). Build a quick endpoint to serve your model
* [1] Containers: [Docker](https://docs.docker.com/get-started/) basics. Can you package your analysis so it runs on my machine?
* [2] Version control: git beyond add/commit/push - feature branches and pull requests

### MLOps & Deployment
* [1] Experiment tracking: [MLflow](https://mlflow.org/) or [Weights & Biases](https://wandb.ai/)
* [2] Data orchestration: [Airflow](https://airflow.apache.org/) or [Dagster](https://dagster.io/) for pipeline scheduling
* [2] Cloud ML platforms: AWS SageMaker, GCP Vertex AI, or Azure ML (pick one, know the concepts)

---

## III. Data Manipulation (SQL & Engineering)
*Replacing "MapReduce / Hive"*

### Modern SQL
* [1] CTEs (`WITH` clauses) - standard now; nested subqueries are a readability red flag
* [1] Window functions: `RANK()`, `LEAD()`, `LAG()` - still the #1 technical screen filter
* [2] Query optimization basics: `EXPLAIN` plans, indexing strategy

### Data Formats & Tools
* [2] Parquet vs. CSV: why row-based is bad for analytics
* [2] JSONL: the standard for LLM fine-tuning datasets
* [2] [Polars](https://pola.rs/) / [DuckDB](https://duckdb.org/) - modern alternatives to pandas for larger-than-memory data

---

## IV. Statistics & Experimental Design
*Replacing "Classical Stats"*

### A/B Testing (Still King)
* [1] Sample size calculation, power analysis, and T-tests
* [1] Metric guardrailing: improve CTR without hurting latency
* [2] Bayesian vs. frequentist A/B testing approaches

### Core Statistical Literacy
* [1] Hypothesis testing, p-values, confidence intervals
* [2] Regression (linear, logistic) and when to use each
* [2] Resampling methods: bootstrap, permutation tests

---

## V. Machine Learning (Classical)
*Replacing the original ML section*

### Tabular ML (Still Dominant for Business Data)
* [1] [XGBoost](https://xgboost.readthedocs.io/) / [LightGBM](https://lightgbm.readthedocs.io/) - for churn, fraud, pricing, these beat neural nets 9/10 times
* [1] Feature engineering and selection for structured data
* [2] Model selection: cross-validation, hyperparameter tuning

### System Design for ML
* [1] "Design a recommendation system" - common interview format
* [2] "Design a RAG pipeline" - the 2026 equivalent
* [2] Cost/latency/quality tradeoffs in production ML

### Specialized (Skip Unless Role-Specific)
* [3] Computer vision, reinforcement learning - too specialized for generalist interviews

---

> **The Key Shift:**
>
> In 2015, the question was *"Can you derive the math?"*
>
> In 2026, the question is *"Can you build a system that uses this math without costing us $1M in API credits?"*
