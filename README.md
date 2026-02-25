# PhD to Data Science Transition Study List

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
Comes up in nearly every technical screen.
* Window functions (`RANK()`, `LEAD()`, `LAG()`, `ROW_NUMBER()`)
* CTEs / `WITH` clauses
* Joins (inner, left, full outer) and aggregations (`GROUP BY`, `HAVING`)
* Query readability: clean formatting, capitalized keywords

### Statistics & Experimentation
The core skill here is judgment: numbers come from distributions, and you need to understand the properties of the distribution you're reasoning about to make sound claims about confidence and comparison.
* Distributional thinking: know what generated your data, what assumptions you're making, and when a summary statistic is (or isn't) enough
* A/B testing end-to-end: hypothesis, sample size, power, test selection, interpretation
* Metric guardrailing: improving one metric without degrading others
* Confidence intervals, bootstrapping, and when parametric vs. non-parametric methods apply
  - You could use bootstrapping for nearly everything here — and that's fine unless data volume or latency says otherwise. The skill is knowing the tradeoffs.

### Classical ML (Tabular Data)
Still the bread and butter for most business problems.
* XGBoost / LightGBM - the default for structured data
* Linear & logistic regression — the math and the assumptions behind them
* Model selection: cross-validation, bias-variance tradeoff
* Feature engineering and selection

### Python Fundamentals
Most screens involve live-coding in Python. R is also widely used in data science — especially in biotech, pharma, and statistics-heavy roles — so if your target industry leans R, invest there too. The concepts below apply in either language.
* Data structures: lists, dicts, sets, and their complexities
* Pandas / Polars for data manipulation
* Writing clean, modular code
  - Notebooks are great for exploration, EDA, and sharing results. The anti-pattern is 800-line notebooks with no functions. When you find yourself copying the same cleaning code between notebooks, that's your signal to refactor it into a `.py` file you can import.

---

## Tier 2: Should Know [!!]

### Generative AI & LLMs
Increasingly expected in 2026 interviews.
* Transformer intuition: what they are, why they fail, how context windows work
  - The architecture behind GPT, Claude, and Gemini. Read one good visual explainer and the "magic" becomes concrete. → [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
* RAG: connecting a vector DB to an LLM (the "Hello World" of applied AI)
  - Your LLM doesn't know your company's internal docs. RAG = search a database for relevant chunks, paste them into the prompt, get grounded answers. It's the most common LLM application pattern in industry. → [RAG from scratch tutorial](https://learnbybuilding.ai/tutorial/rag-from-scratch/)
* Prompt engineering: system prompts, few-shot, structured outputs
  - "System prompt" = persistent instructions (e.g., "You are a medical assistant"). "Few-shot" = giving 2-3 examples in the prompt so the model follows the pattern. "Structured output" = asking the model to respond in JSON. → [Prompt Engineering Guide](https://www.promptingguide.ai/)
* Evaluation: golden datasets, semantic similarity, exact match
* Fine-tuning vs. prompting: when to use each, cost tradeoffs

### Software Engineering
Complements your research skills and is very learnable. Most of this is pattern recognition, not new theory.
* Production code: type hints, testing, clean structure
  - Type hints: e.g., `def predict(features: pd.DataFrame) -> np.ndarray:` — tells your editor and teammates what goes in and comes out. Your IDE will catch bugs before you run anything. → [Python typing docs](https://docs.python.org/3/library/typing.html)
  - Testing: you write `assert clean_text("  HELLO ") == "hello"` in a test file, run `pytest`, and it tells you if anything broke. This is how teams catch bugs before they ship. → [pytest getting started](https://docs.pytest.org/en/stable/getting-started.html)
* APIs: FastAPI basics — serve a model behind an endpoint
  - You decorate a function with `@app.post("/predict")`, and now your model is a URL that any app can send data to and get predictions back. This is how models get used in the real world. → [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/)
* Docker: package your work so it runs anywhere
  - Useful to understand conceptually, but rarely tested in interviews — you'll learn this on the job. → [Docker for Data Science](https://www.datacamp.com/tutorial/docker-for-data-science-introduction)
* Git: branches, PRs, not just add/commit/push
  - In a team, you don't push directly to `main`. You create a branch, make changes, open a Pull Request, teammates review your code, then it gets merged. This is how every company works. → [GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow)

### Algorithms & Data Structures
Less weight than 2016, but still appears in technical screens.
* Big-O notation and common complexities
  - How fast does your code get when the data gets 100x bigger? Looking up a value in a list is O(n) — you check every item. In a dict it's O(1) — instant. Interviewers want to hear you reason about scale.
* Sorting: know merge sort and quicksort cold
* Search: binary search, BFS, DFS
* Dynamic programming basics (memoization, common patterns)

### Probability & Bayes
Comes up in product-sense and ML questions alike.
* Bayes' theorem and conditional probability
* Common puzzles: disease testing, Monty Hall
* Probability distributions and when to apply them

### Product Sense & Business Analytics
The skill of translating a vague business question into a concrete data science problem. Heavily tested at product-focused companies.
* Problem framing: metric definition, segmentation, experiment design
  - A PM says "engagement is down." You ask: down for whom? Measured how? Compared to what baseline? This framing skill *is* the interview.
  - Practice taking vague business problems and defining: the metric, the segmentation, the model/experiment, and what you'd recommend if results are ambiguous.
* Funnel analysis, cohort analysis, churn, LTV
* Correlation vs. causation in business context

---

## Tier 3: Nice to Know [!]

### MLOps & Deployment
Differentiates "I can build a model" from "I can ship a model."
* Experiment tracking: MLflow or Weights & Biases
  - Think of it as a lab notebook for ML. Every time you train a model, it logs the hyperparameters, metrics, and artifacts so you can compare 50 runs and reproduce the best one. → [MLflow quickstart](https://mlflow.org/docs/latest/getting-started/intro-quickstart/)
* Pipeline orchestration: Airflow or Dagster
  - "Every morning at 6am: pull new data → retrain model → run tests → deploy if metrics improve." That workflow, automated and monitored, is a pipeline. → [Dagster getting started](https://docs.dagster.io/getting-started)
* Cloud ML: SageMaker / Vertex AI / Bedrock (pick one, know concepts)

### System Design for ML
Increasingly common in senior-level loops.
* "Design a recommendation system"
* "Design a RAG pipeline"
* Cost / latency / quality tradeoffs

### Modern Data Formats & Tools
Shows you're current.
* Parquet vs. CSV, JSONL for LLM datasets
  - Parquet is a columnar, compressed format. A 2GB CSV might become ~200MB Parquet and load 10x faster in pandas. If you're still using CSV for anything over 100MB, switch. → [Apache Parquet overview](https://parquet.apache.org/docs/overview/)
* DuckDB for local analytics
  - Run SQL directly on local files with zero setup: `SELECT * FROM 'data.parquet' WHERE revenue > 1000`. No database server needed. → [DuckDB tutorial (MotherDuck)](https://motherduck.com/blog/duckdb-tutorial-for-beginners/)
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
