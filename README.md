# PhD to Data Science Study Guide

Study materials for PhDs transitioning into data science, organized by **importance to interview success**. This is a study guide — it tells you *what to learn*, ranked by how likely it is to come up in interviews. It is not an interview strategy guide or a career transition roadmap.

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

## The Interview Pipeline

Understanding what you're preparing for:

1. **Application:** Either a cold application or a referral. For a concrete starting point, find a data science job posting you're interested in and use it to identify your gaps against this guide.
2. **Behavioral screen:** Are you someone others want to work with? Can you navigate conflict? Are you aligned with the company's values and genuinely excited about the role? *Behavioral prep is important but out of scope for this guide — it's its own skill set.*
3. **Technical screens (1-2):** These test a skill you must be proficient in to do the job. If the role requires Python, expect to solve a problem in Python. If it requires SQL, expect a SQL screen.
4. **Take-home challenge:** Fairly common. You work on an assessment independently over a longer period and submit it — think of it like a tightly scoped Kaggle challenge. You may then present your results to a panel that asks about your design choices.
5. **On-site interviews (4-5 rounds, ~45 min each):** These test your skills across different areas. Expect some combination of: SQL problem, programming problem, business metric / system design problem, behavioral / interpersonal problem, and machine learning problem. Formats vary — case studies, live coding, whiteboarding, etc.
6. **Negotiation:** The hiring manager expresses strong interest. You negotiate terms — salary, equity, vacation, start date. Don't skip this step.
7. **Offer:** You sign a document accepting the terms.

**A note on environment:** In live technical screens, you often won't have access to an IDE — it might be a Google Doc, CoderPad, or a whiteboard. No autocomplete, no keyboard shortcuts. Practice solving problems in a plain text environment so you're not thrown off when it happens.

---

## Tier 1: Must Know [!!!]

### SQL
Comes up in nearly every technical screen.
* Window functions: `RANK()`, `DENSE_RANK()`, `ROW_NUMBER()`, `LEAD()`, `LAG()`, `NTILE()`, running totals with `SUM() OVER`
  - These let you compute values across rows without collapsing them — e.g., "rank each user's purchases by date" or "compare each row to the previous one." This is the #1 topic in SQL screens. → [Mode: SQL window functions](https://mode.com/sql-tutorial/sql-window-functions)
* CTEs / `WITH` clauses for readable, composable queries
  - Named subqueries you define at the top, then reference like tables. They turn a nested mess into something you can read top-to-bottom. → [SQLBolt interactive tutorial](https://sqlbolt.com/)
* Joins (inner, left, right, full outer, cross, self-join) and aggregations (`GROUP BY`, `HAVING`, `COUNT(DISTINCT ...)`)
  - → [Visual guide to SQL joins](https://www.atlassian.com/data/sql/sql-join-types-explained-visually)
* Subqueries: correlated vs. uncorrelated, `EXISTS` vs. `IN`
  - → [Mode: SQL subqueries](https://mode.com/sql-tutorial/sql-sub-queries)
* `CASE WHEN` for conditional logic, `COALESCE` / `NULLIF` for null handling
  - → [W3Schools: SQL CASE](https://www.w3schools.com/sql/sql_case.asp)
* Date/time manipulation: `DATE_TRUNC`, `EXTRACT`, `INTERVAL`, timezone awareness
  - → [PostgreSQL date/time functions](https://www.postgresql.org/docs/current/functions-datetime.html)
* Query readability and performance: clean formatting, indexing intuition, avoiding `SELECT *` in production
  - → [How database indexing works](https://www.atlassian.com/data/sql/how-indexing-works)

### Statistics & Experimentation
The core skill here is judgment: numbers come from distributions, and you need to understand the properties of the distribution you're reasoning about to make sound claims about confidence and comparison.
* Distributional thinking: know what generated your data, what assumptions you're making, and when a summary statistic is (or isn't) enough
  - Central limit theorem, law of large numbers, normal / log-normal / Poisson / binomial — know which applies to your data and why it matters for your choice of test or interval. → [Seeing Theory: probability distributions](https://seeing-theory.brown.edu/probability-distributions/index.html)
* A/B testing end-to-end: hypothesis formulation, sample size calculation, statistical power, test selection, interpretation
  - This is the full workflow: define a metric, decide how much lift you care about, calculate how many samples you need, run the experiment, interpret the result. Companies run hundreds of these. → [Evan Miller: A/B testing guide](https://www.evanmiller.org/how-not-to-run-an-ab-test.html)
* Metric guardrailing: improving one metric without degrading others
  - e.g., you optimize click-through rate but need to verify you haven't tanked revenue or increased support tickets. Multi-metric thinking is expected. → [PostHog: guardrail metrics](https://posthog.com/product-engineers/guardrail-metrics)
* Confidence intervals, bootstrapping, and when parametric vs. non-parametric methods apply
  - You could use bootstrapping for nearly everything here — and that's fine unless data volume or latency says otherwise. The skill is knowing the tradeoffs. → [Khan Academy: confidence intervals](https://www.khanacademy.org/math/statistics-probability/confidence-intervals-one-sample)
* Effect size, multiple comparisons (Bonferroni, FDR), and Simpson's paradox
  - Statistical significance alone isn't enough — how big is the effect? And when you run many tests, some will be "significant" by chance. These are the gotchas interviewers probe. → [Statistics By Jim: effect size](https://statisticsbyjim.com/hypothesis-testing/effect-size/)

### Classical ML (Tabular Data)
Still the bread and butter for most business problems.
* XGBoost / LightGBM — the default for structured/tabular data
  - Gradient-boosted decision trees. They win most Kaggle competitions on tabular data for a reason: fast, handle missing values, capture nonlinear relationships. Know how to tune `learning_rate`, `max_depth`, `n_estimators`, and when you're overfitting. → [XGBoost getting started](https://xgboost.readthedocs.io/en/stable/get_started.html)
* Linear & logistic regression — the math and the assumptions behind them
  - Linear: continuous outcome, assumes linear relationship, interpretable coefficients. Logistic: binary outcome, outputs probabilities via sigmoid. Know what multicollinearity does, what regularization (L1/L2) is for, and when these simple models outperform complex ones. → [Google ML Crash Course: logistic regression](https://developers.google.com/machine-learning/crash-course/logistic-regression)
* Model selection: cross-validation, bias-variance tradeoff, train/validation/test splits
  - Why you never evaluate on training data. K-fold cross-validation gives you a more honest estimate of generalization. Bias-variance tradeoff is the tension between underfitting (too simple) and overfitting (too complex). → [Scikit-learn: cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
* Feature engineering and selection
  - Encoding categoricals (one-hot, target encoding), handling missing values, creating interaction terms, log transforms for skewed features. Feature importance via permutation or SHAP values. → [Feature engineering guide (Kaggle)](https://www.kaggle.com/learn/feature-engineering)
* Evaluation metrics: accuracy, precision, recall, F1, AUC-ROC, log loss
  - Which metric matters depends on the problem. Fraud detection with 0.1% fraud rate? Accuracy is useless — you need precision/recall. Know the confusion matrix cold. → [Scikit-learn: model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

### Python Fundamentals
Most screens involve live-coding in Python. R is also widely used in data science — especially in biotech, pharma, and statistics-heavy roles — so if your target industry leans R, invest there too. The concepts below apply in either language.
* Data structures: lists, dicts, sets, tuples, and their complexities
  - Know when to reach for each. Dicts and sets are O(1) lookup (hash-based); lists are O(n) for search. Defaultdicts, Counter, namedtuple, and deque come up in coding screens. → [Python docs: data structures](https://docs.python.org/3/tutorial/datastructures.html)
* Pandas / Polars for data manipulation
  - groupby, merge, pivot, apply, vectorized operations vs. iterrows (don't). Polars is gaining traction for speed on larger datasets. Know how to chain operations and avoid SettingWithCopyWarning. → [Pandas getting started tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/)
* List comprehensions, generators, lambda functions, `map`/`filter`
  - Concise, idiomatic Python. Generators are important when your data doesn't fit in memory — they yield one item at a time. → [Python Like You Mean It: generators & comprehensions](https://www.pythonlikeyoumeanit.com/Module2_EssentialsOfPython/Generators_and_Comprehensions.html)
* Writing clean, modular code
  - Notebooks are great for exploration, EDA, and sharing results. The anti-pattern is 800-line notebooks with no functions. When you find yourself copying the same cleaning code between notebooks, that's your signal to refactor it into a `.py` file you can import. → [Real Python: structuring Python projects](https://realpython.com/python-application-layouts/)

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
  - How do you know if your LLM app is working? Build a test set of input/expected-output pairs, then measure how close the model gets. → [Confident AI: LLM evaluation metrics](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
* Fine-tuning vs. prompting: when to use each, cost tradeoffs
  - Prompting is fast and cheap to iterate. Fine-tuning is expensive but can improve quality for narrow tasks. Most production systems start with prompting and only fine-tune when they hit a ceiling. → [Google ML Crash Course: LLM tuning](https://developers.google.com/machine-learning/crash-course/llm/tuning)

### Software Engineering
Complements your research skills and is very learnable. Most of this is pattern recognition, not new theory. Some of these are tested in interviews; others are day-to-day skills you'll pick up on the job. Both are listed here because understanding them helps you talk credibly about how you work.
* Production code: type hints, testing, clean structure
  - Type hints: e.g., `def predict(features: pd.DataFrame) -> np.ndarray:` — tells your editor and teammates what goes in and comes out. Your IDE will catch bugs before you run anything. → [Python typing docs](https://docs.python.org/3/library/typing.html)
  - Testing: you write `assert clean_text("  HELLO ") == "hello"` in a test file, run `pytest`, and it tells you if anything broke. This is how teams catch bugs before they ship. → [pytest getting started](https://docs.pytest.org/en/stable/getting-started.html)
* APIs: FastAPI basics — serve a model behind an endpoint
  - You decorate a function with `@app.post("/predict")`, and now your model is a URL that any app can send data to and get predictions back. This is how models get used in the real world. → [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/)
* Docker: package your work so it runs anywhere *(day-to-day skill — rarely tested in interviews)*
  - Useful to understand conceptually. You'll learn the details on the job. → [Docker for Data Science](https://www.datacamp.com/tutorial/docker-for-data-science-introduction)
* Git: branches, PRs, not just add/commit/push *(day-to-day skill — expected but rarely tested directly)*
  - In a team, you don't push directly to `main`. You create a branch, make changes, open a Pull Request, teammates review your code, then it gets merged. → [GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow)

### Algorithms & Data Structures
Less weight than 2016, but still appears in technical screens.
* Big-O notation and common complexities
  - How fast does your code get when the data gets 100x bigger? Looking up a value in a list is O(n) — you check every item. In a dict it's O(1) — instant. Interviewers want to hear you reason about scale. → [samwho: Big-O notation (visual)](https://samwho.dev/big-o)
* Sorting: know merge sort and quicksort cold
  - → [VisuAlgo: sorting algorithms](https://visualgo.net/en/sorting)
* Search: binary search, BFS, DFS
  - → [VisuAlgo: graph traversal (BFS/DFS)](https://visualgo.net/en/dfsbfs)
* Dynamic programming basics (memoization, common patterns)
  - → [freeCodeCamp: demystifying dynamic programming](https://www.freecodecamp.org/news/demystifying-dynamic-programming-3efafb8d4296/)

### Probability & Bayes
Comes up in product-sense and ML questions alike.
* Bayes' theorem and conditional probability
  - → [BetterExplained: intuitive Bayes' theorem](https://betterexplained.com/articles/an-intuitive-and-short-explanation-of-bayes-theorem/)
* Common puzzles: disease testing, Monty Hall
* Probability distributions and when to apply them
  - → [Seeing Theory: probability distributions](https://seeing-theory.brown.edu/probability-distributions/index.html)

### Product Sense & Business Analytics
The skill of translating a vague business question into a concrete data science problem. Heavily tested at product-focused companies.
* Problem framing: metric definition, segmentation, experiment design
  - A PM says "engagement is down." You ask: down for whom? Measured how? Compared to what baseline? This framing skill *is* the interview.
  - Practice taking vague business problems and defining: the metric, the segmentation, the model/experiment, and what you'd recommend if results are ambiguous. → [DataLemur: product sense interview questions](https://datalemur.com/blog/product-sense-interview-questions)
* Funnel analysis, cohort analysis, churn, LTV
  - → [Chartio: cohort analysis primer](https://chartio.com/learn/marketing-analytics/cohort-analysis-primer/)
* Correlation vs. causation in business context
  - → [Scribbr: correlation vs. causation](https://www.scribbr.com/methodology/correlation-vs-causation/)

---

## Tier 3: Nice to Know [!]

### MLOps & Deployment
These are mostly day-to-day skills rather than interview topics. Knowing they exist and being able to discuss them shows maturity, but you'll learn the tools on the job.
* Experiment tracking: MLflow or Weights & Biases
  - Think of it as a lab notebook for ML. Every time you train a model, it logs the hyperparameters, metrics, and artifacts so you can compare 50 runs and reproduce the best one. → [MLflow quickstart](https://mlflow.org/docs/latest/getting-started/intro-quickstart/)
* Pipeline orchestration: Airflow or Dagster
  - "Every morning at 6am: pull new data → retrain model → run tests → deploy if metrics improve." That workflow, automated and monitored, is a pipeline. → [Dagster getting started](https://docs.dagster.io/getting-started)
* Cloud ML: SageMaker / Vertex AI / Bedrock (pick one, know concepts)
  - → [AWS vs Azure vs GCP for ML (DataCamp)](https://www.datacamp.com/blog/aws-vs-azure-vs-gcp)

### System Design for ML
Increasingly common in senior-level loops.
* "Design a recommendation system"
* "Design a RAG pipeline"
* Cost / latency / quality tradeoffs
* → [Exponent: ML system design interview guide](https://www.tryexponent.com/blog/machine-learning-system-design-interview-guide)

### Modern Data Formats & Tools
Shows you're current.
* Parquet vs. CSV, JSONL for LLM datasets
  - Parquet is a columnar, compressed format. A 2GB CSV might become ~200MB Parquet and load 10x faster in pandas. If you're still using CSV for anything over 100MB, switch. → [Apache Parquet overview](https://parquet.apache.org/docs/overview/)
* DuckDB for local analytics
  - Run SQL directly on local files with zero setup: `SELECT * FROM 'data.parquet' WHERE revenue > 1000`. No database server needed. → [DuckDB tutorial (MotherDuck)](https://motherduck.com/blog/duckdb-tutorial-for-beginners/)
* LangChain / LlamaIndex for LLM orchestration
  - → [LangChain tutorials](https://python.langchain.com/docs/tutorials/)

### Responsible AI & Ethics
Asked more often than candidates expect.
* Hallucination mitigation and guardrails
  - → [Anthropic: reducing hallucinations](https://docs.anthropic.com/en/docs/build-with-claude/reduce-hallucinations)
* Bias detection in models and datasets
  - → [Google ML Crash Course: fairness](https://developers.google.com/machine-learning/crash-course/fairness)
* When *not* to use ML
  - → [Google ML guide: problem framing](https://developers.google.com/machine-learning/problem-framing)

---

## Interview Mindset

Interviews vary widely — some are deeply technical, some are conversational, some are case studies. The common thread is that interviewers want to see how you think through problems, not just whether you know the answer.

**Preparation:**
* Know your resume and projects cold. For each project, be ready to explain: the business problem, your approach, what you'd do differently, and the impact.
* Research each company before the interview. What data problems do they solve? What would you build if you worked there? Having a specific opinion shows you've done the work.
* Practice communicating technical work to non-technical audiences. "I built a gradient-boosted model" means nothing to a PM — "I built a model that predicts which users will churn next month, so we can intervene early" does.

**During the interview:**
* Break problems down and narrate as you go. When writing code live, show how you decompose a big problem into smaller pieces — write a helper function, explain what it does and why, then use it. The ability to structure your thinking in real time is what interviewers are evaluating.
* Think out loud. Say what you're considering, what tradeoffs you see, and why you're making each choice. The reasoning matters more than the final answer.
* It's OK to say "I don't know, but here's how I'd approach figuring it out." Intellectual honesty is valued over bluffing.
* Ask clarifying questions. Ambiguous prompts are often intentional — demonstrating that you scope a problem before diving in is itself the skill being tested.

**Resources:**
* [Ace the Data Science Interview (book)](https://www.acethedatascienceinterview.com/) — comprehensive question bank covering SQL, ML, stats, and product sense
* [DataLemur](https://datalemur.com/) — free SQL and data science interview practice
* [Lean Analytics (book)](https://leananalyticsbook.com/) — how to think about metrics and business problems, useful for product sense
