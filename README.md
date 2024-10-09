# Introduction  

This repository contains materials I generated as part of my preparations for interviews in insight data science!

Within this repository, I have ranked the study topics, and will create python notebooks or links to solutions, and further study.

Topics (overall) will be rated with a score (1, 2, or 3), where 3 is "very prepared", 2 is "familiar, needs study" and 3 is "not at all prepared".

 
## Effective Communicaiton [3]

This needs to be practiced while studying.

Reading List: Sell Yourself in any Interview, by Oscar Adler, Chapter 1, 2

## CS Fundamentals 
### Resources
* [1] [VisualAlgo](http://visualgo.net/), visualizing algorithms
* [3] [Data Types, Operators, Variables](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-00-introduction-to-computer-science-and-programming-fall-2008/video-lectures/lecture-1)
* [3] [Branching, Conditionals, Iteration](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-00-introduction-to-computer-science-and-programming-fall-2008/video-lectures/lecture-2)
* [2] [Common Code Patterns, Iterative Programs](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-00-introduction-to-computer-science-and-programming-fall-2008/video-lectures/lecture-3)
* [2] [Abstraction through Functions, Recursion](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-00-introduction-to-computer-science-and-programming-fall-2008/video-lectures/lecture-4)
* [2] [Floating point numbers, root finding](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-00-introduction-to-computer-science-and-programming-fall-2008/video-lectures/lecture-5)
* [2] [Bisection Methods, Newton/Raphson, Intro to Lists] (http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-00-introduction-to-computer-science-and-programming-fall-2008/video-lectures/lecture-6)
* [3] [Lists and Mutability, Dictionaries, Intro to Efficiency](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-00-introduction-to-computer-science-and-programming-fall-2008/video-lectures/lecture-7)
* [1] [Complexity, Log, Linear, Quadratic, Exponential Algorithms](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-00-introduction-to-computer-science-and-programming-fall-2008/video-lectures/lecture-8)
* [1] [Binary Search, Bubble and Selection Search](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-00-introduction-to-computer-science-and-programming-fall-2008/video-lectures/lecture-9)
* [1] [Divide and Conquor Methods, Merge Sort, Exceptions](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-00-introduction-to-computer-science-and-programming-fall-2008/video-lectures/lecture-10)

**More Advanced CS**
* [1] [Code the problems, solutions, from scratch, lectures 11-24](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-00-introduction-to-computer-science-and-programming-fall-2008/video-lectures/)
* [1] [Python - Algorithms, Data Structures](http://interactivepython.org/runestone/static/pythonds/index.html)
* [2] [Leetcode programming challenge problems](https://leetcode.com/)

### CS Topics
* [3] $\mathcal{O}(n)$ notation, and complexity.
* **Data Structures**
    * [2] Hash Table
    * [1] Stack/Queue/Deque
    * [3] LinkedList 
    * [1] Heap - constructs in $\mathcal{O}(n)$ time
    * [2] Trees - balanced - same # nodes on each side for level n-1, vs unbalanced
* ** Recursion and Master Theorem **
* ** Algorithms [Coursera Link](https://www.coursera.org/login?course_id=970400&r=https%3A%2F%2Fclass.coursera.org%2Falgs4partI-003&user_action=class&topic_name=Algorithms%2C%20Part%20I) **
    * **Sorting Algorithms**
        * [3] Bubble Sort, $\mathcal{O}(n^2)$ complexity, constant memory
        * [3] Selection Sort, $\mathcal{O}(n^2)$ complexity, constant memory
        * [3] Insertion Sort, $\mathcal{O}(n^2)$ complexity, constant memory
        * [3] Merge Sort, $\mathcal{O}(n log(n))$ complexity, 2n memory
        * [2] Quick Sort, $\mathcal{O}(n^2)$ complexity, constant memory
        * [2] Heap Sort, $\mathcal{O}(n log(n))$ complexity, constant memory
        * [2] Timsort, (hybrid merge/insertion) (just need to know this is how python works internally, with sorted)
    * **Graph Algorithms**
        * [2] DFS (Depth-First Search)
        * [2] BFS (Breadth-First Search)
        * [1] Dijkstra's Search (Shortest Path)
        * [1] Path of least resistence
    * **Dynamic Programming **
        * [1] Memorization
        * [1] Knapsack Problem
        * [1] Fibonacci Sequence
    * **Resevoir Sampling**
* **Math and Brain Teasers**
    * [3] Combinatorics:
        * $_nC_k$ style problems
        * "two eggs" problem
        * five fastest horses problem
* **Sample Problems**
    * Write a function that lists all three digit lock combiniations, each number $0\lt n\leq30$, non-repeating. Generalize to more than three numbers, with no local storage.
    * [2] Traverse binary search tree in order without recursion
    * [2] Write a function which takes a base-10 number and postive integer, $k_i \leq 2 $ and print out number in base k 
    * [1] Solve the 0-1 knapsack problem
    * [2] Given an array of numbers, find the subarray which has the greatest sum, $\mathcal{O}(n)$
* ** Text Processing Problems ** 
    * [2] Replace a string in-place with no extra memory
    * [3] Remove stop words from a sentence
    * [3] Find palendromes in any string
 

## SQL + MapReduce & Hive

### Resources
* [Mode SQL School](http://sqlschool.modeanalytics.com/)
* [Coursera SQL Class](https://class.coursera.org/db/quiz/index)

### Mechanics
* Joins:
    * [3] Inner
    * [3] Left (outer)
    * [3] Right (outer)
    * [3] **Full Outer** **[Important]**
* Selects:
    * [2] Nested Select
    * [2] Create as select
    * [1] Views
* [3] As, Where, Order By, Limit
* [1] Aggregate functions
    * [3] Having
    * **[3] Group By** **[Important]**
    * [3] min/max, avg (mean), count, first, last
    * [2] Insert/delete
    * [1] union
    * [1] window
    * [1] If/Then
        * [3] Case Statements
        * [2] Coalesce
        * [1] IF in SELECT
        

### General Knowlege
* [2] Pros, Cons of SQLite, MySQL, PostgreSQL, Oracle, NoSQL, etc
* [1] Principals of MapReduce/Hadoop/Hive, Shard, RethinkDB
* [3] Style:
    * Show that you write clean SQL - use proper indentation
    * Capitalize keywords, even if mid word. TableNames, Variables are lowercase

## Analytics

### Resoruces
Lean Analytics (see dropbox book)

### Analytics Genres
* Open Ended Questions
    * [1] For each company you have an interview with, do a deep product dive to answer the following questions:
        * What are the major problems, especially related to data, that the company seeks to solve?
        * If I was working at the company, what problems would I want to work on, what products/features would I build?
        * How can I improve the company's product?
        * What would be the primary challenges in building the product?
* [1] Product Analytics
    * How do we measure success?
    * "Give me every conclusion you can make from this graph (no labels, etc)"
    * Deep dive into product
* [1] Business Analytics
* [1] Marketing Analytics
    * [1] SEO
    * [1] SEM
    * [1] Social Marketing
    * [1] Viral Coefficient
    
### Analytics Key Concepts
* [1] Churn
* [1] Customer Lifetime Value (LTV)
* [1] Funnel
* [1] Cohort Analysis
* [1] Correlation vs Causation
* [1] A/B Testing
    * A/A Testing
    * Control Group
    * [1] Stats
        * Distribution Tests
        * Power Analysis
        * KS-test
* [1] Experimental Design
    * Give a scenario, describe what I would build.


## Statistics

Hiring Managers expect Insight Fellows to have very good statistics backgrounds. This section should be given a high priority.

### Resources
[Khan Academy Video Series](https://www.khanacademy.org/math/probability)

### Statistics Key Concepts
* [2] T-Test **[Important]**
    * difference between "one failed" and "two failed" tests
    * [2] ANOVA (requirements: independance, approximately normal constant varience) **[Important]**
        * Simple use cases for ANOVA
* [2]Probability, and [Bayes Theorem](http://georgemdallas.wordpress.com/2013/07/13/how-to-build-an-anti-aircraft-missile-probability-bayes-theorem-and-the-kalman-filter/) **[Important]**
    * [2] p value
    * [1] Monty Hall Problem
    * [2] What's the probability you have a disease if you test positive for it??
* [2] Modeling Regressions **[Important]**
    * [2] multiple regression
    * [1] logistic regression
    * [1] feature selection
* [1] Hypothesis Testing (complicated..?) **[Important]** 
* [1] $\chi^2$ test for normality
    * When is it appropriate to assume a normal distribution, when is it not? **[Important]**
* [3] $R^2$ (Pearson's correlation coeffienent, squared) **[Important]**
* [1] QQ Plot
* [1] "descriptave statistics"
* [2] Maximum likelihood estimator
* [1] Kolmogorov-Smirinov tests (non parametric)
* [1] [Power Analysis] (http://www.ats.ucla.edu/stat/seminars/Intro_power/)
* [1] [Bonferroni Correction](http://en.wikipedia.org/wiki/Bonferroni_correction)
* Use cases for different statistics (mean, median, standard deviation, varience, standard error, etc)
    * [2] Confidence Intervals
    * [1] Type 1 and Type 2 error
        * OIStats (p 176)
            * Type 1 Error is rejecting the null hypothesis when H0 is actually true
            * Type 2 Error is failing to reject the null hypothesis when the alternative is actually true.
    * [3] When to use mean instead of median? When do we need to arithmetically calculate the average?
    * [1] Sampling distributions
    * [1] Resampling
        * [2] bootstrap (random sampling) 
        * [2] bootstrap for confidence interval on median
        * [2] jack-knife (all possible subsets)
        
### Sample Statistics Questions
* [1] How would you do variable selection? What kind of model with x data? Can you explain what hypothesis testing means? What is a p-value?
* [1] How do you know if the data you have is appropriate to model in a regression?
* [1] How do you test it?
* [1] How do you know if your data is normally distriubted?
* [1] What do you do if your data is not normally distributed?
* [1] How would you test if your data is distriubted the same as another system?

## Machine Learning

### Resources
**Course:** [Machine Learning by Andrew Ng](http://cs229.stanford.edu/materials.html)

**Blog** [Shape of Data](http://shapeofdata.wordpress.com/) 

**Github** [Learn Data Science](http://nborwankar.github.io/LearnDataScience/) 

### Topics

* [2] Linear Regression **[Important]**
    * [1] Regularization, number of features
    * [2] Know derivation/algorithm (cost function?? Gradient Descent, etc?)
* [1] Logistic Regression **[Important]** 
* [2] Optimization Schemes: Gradient Descent, Conjugate Gradient, BFGS
* [2] Random Forests (decision trees) **[Important]**
* [2] Clustering -- kmeans (k-nearest-neighbors) **[Important]**
* [1] Model Selection (in particular, k-fold validation) **[Important]**
* [2] SVM (support vector machines)
* [1] [Neural Networks](https://www.coursera.org/course/neuralnets)
* [1] SVD, PCA, and [Kohonen Maps](http://en.wikipedia.org/wiki/Self-organizing_map)
    * Don't name componants, they are an arbitrary combination of actual variables.
* [1] Recommendation algorithms, see python-recsys, crab framework, etc
* [3] Naive Bayes
* [1] Python - sci-kit learn, [Apache Mahout](http://mahout.apache.org/)
* Know use case, pros and cons, of each algorithm
* Get to know algorithms behind others' projects



## Tools For Data Challenges

### Python Tools
* [3] Try to learn python "the hard way"
* [3] ipython notebooks
* [3] pip install/freeze
* Packages
    * Numerical Analysis
        * [1] numpy, scipy, [pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html)
        * 6 hours lecture on pandas: [1](http://www.youtube.com/watch?v=w26x-z-BdWQ), [2](http://www.youtube.com/watch?v=0unf-C-pBYE)
    * Plotting:
        * [1] pylab/matplotlib, vincent, daft
    * [1] machine learning: SciKit learn, recsys, crab
    * [1] data science toolkit [dstk](http://www.datasciencetoolkit.org/)
    * [1] database/storage: [dataset](https://dataset.readthedocs.org/en/latest/), MySQLdb, sqlite, shove, shelve, pickle/cPickle
    * [3] web: urllib/httplib, requests, scrapy
* Clean code: use google python style guide
* R For statistical analysis
    * plotting: ggplot2, lattice
    * Rpy
* Web Dev
    * [2] Flask, gunicorn, Twistd, HTML/CSS, Javascript
    * Handy modules
        * [2] Twitter Bootstrap
        * [1] d3.js
        * [2] jQuery
        * [1] React
        * [1] purecss
        * [1] moment.js
        * [1] leaflet.js
        * [1] meteor.js/angular.js/ember.js/backbone.js
        * [2] AWS
        * [2] EC2
        * [2] DNS
    * Presentation Sharing - eh, who cares
    * Github
    * Linux (duh)
    * Mapping Tools
        * Google maps api
        * folium
        * leaflet.js
        * kartograph
        * PostGIS
        * Grass GIS
        * GMT        

## Misc Notes

Make sure ot know project, resume inside and out. Resume - algorithm names, tools. Lookup background of interviewers, check for background. Speak on project with conviction.
