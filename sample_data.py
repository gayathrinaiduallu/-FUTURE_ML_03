"""
sample_data.py – Simulated resumes and job descriptions for demo/testing.
Each candidate's profile is inspired by realistic hiring scenarios.
"""

# ──────────────────────────────────────────────────────────────
#  JOB DESCRIPTIONS
# ──────────────────────────────────────────────────────────────

JOB_DESCRIPTIONS = {
    "data_scientist": """
We are hiring a Data Scientist to join our AI team.

Requirements:
- Strong programming skills in Python and SQL
- Experience with machine learning frameworks: scikit-learn, TensorFlow, PyTorch
- Proficiency in data analysis using pandas, numpy
- Knowledge of deep learning and neural networks
- Experience with NLP (natural language processing)
- Familiarity with data visualization tools: matplotlib, seaborn, tableau
- Version control with Git
- Cloud experience (AWS or GCP) is a plus
- Strong communication and teamwork skills

Responsibilities:
- Build and deploy ML models
- Perform exploratory data analysis
- Collaborate with product and engineering teams
- Document and communicate findings clearly
""",

    "ml_engineer": """
Machine Learning Engineer – Product Team

We need an ML Engineer experienced in production ML systems.

Must-have skills:
- Python, Scala, or Java
- PyTorch or TensorFlow for model development
- MLOps, CI/CD pipelines for model deployment
- Docker, Kubernetes for containerisation
- AWS or Azure cloud platforms
- SQL and NoSQL databases (PostgreSQL, MongoDB)
- Strong understanding of statistics and linear algebra
- Git, agile, and scrum methodologies

Nice to have:
- Spark, Hadoop for big data
- Hugging Face transformers
- Leadership and communication
""",

    "frontend_developer": """
Frontend Developer – Web Platform

Required Skills:
- React, Angular, or Vue.js frameworks
- HTML, CSS, JavaScript, TypeScript
- REST API integration
- Git version control
- Responsive design and cross-browser compatibility
- Node.js knowledge is a bonus
- Good communication and teamwork
- Agile/Scrum experience preferred
""",
}

# ──────────────────────────────────────────────────────────────
#  SAMPLE CANDIDATE RESUMES
# ──────────────────────────────────────────────────────────────

SAMPLE_RESUMES = {
    "Alice Johnson": """
Alice Johnson
Email: alice@example.com | LinkedIn: linkedin.com/in/alicejohnson

SUMMARY
Data professional with 3 years of experience in machine learning and data analysis.
Passionate about building intelligent systems and communicating insights clearly.

SKILLS
Programming: Python, R, SQL
ML/AI: scikit-learn, TensorFlow, PyTorch, deep learning, neural network
Data: pandas, numpy, matplotlib, seaborn, data analysis, data visualization
Cloud: AWS, GCP
Tools: Git, Jupyter Notebook, tableau
Soft Skills: communication, teamwork, problem solving

EXPERIENCE
Data Scientist – TechCorp (2022–Present)
- Built customer churn prediction model using scikit-learn (AUC: 0.91)
- Deployed NLP pipeline for sentiment analysis using BERT
- Created Tableau dashboards consumed by 50+ stakeholders

EDUCATION
B.Tech Computer Science – IIT Bangalore (2022)

PROJECTS
- Resume Screening System using TF-IDF and cosine similarity
- Sentiment Analysis on Twitter data using NLTK and spaCy
""",

    "Bob Smith": """
Bob Smith
Email: bob@example.com

PROFILE
Software developer transitioning into data science. Comfortable with Python
and basic ML concepts. Eager to learn and grow.

TECHNICAL SKILLS
- Python, Java
- pandas, numpy
- Basic machine learning
- SQL
- Git
- HTML, CSS (hobby projects)
- communication, teamwork

EDUCATION
BSc Computer Science – Delhi University (2023)

WORK EXPERIENCE
Junior Developer – StartupXYZ (2023–Present)
- Maintained backend services in Java
- Wrote SQL queries for reporting dashboards

PERSONAL PROJECTS
- Iris flower classification using scikit-learn
- Simple data visualisation with matplotlib
""",

    "Carol Martinez": """
Carol Martinez – Data & ML Enthusiast
carol.martinez@gmail.com

ABOUT ME
Recent graduate specialising in machine learning and NLP. Published a research
paper on transformer-based text classification.

SKILLS
Languages: Python, R, Scala
Deep Learning: PyTorch, TensorFlow, Keras, Hugging Face, transformers, BERT
NLP: natural language processing, spaCy, NLTK, text classification
Data: pandas, numpy, matplotlib, seaborn, plotly
Big Data: Spark, Hadoop
Cloud: AWS, Azure
MLOps: Docker, Kubernetes, CI/CD, mlops
Databases: PostgreSQL, MongoDB, SQL
Productivity: Git, Agile, Scrum
Soft: communication, leadership, critical thinking, problem solving

EDUCATION
M.Tech Artificial Intelligence – IISC Bangalore (2025)
- Thesis: "Efficient Fine-Tuning of LLMs for Low-Resource NLP Tasks"

EXPERIENCE
ML Research Intern – AI Labs (2024)
- Fine-tuned GPT model for domain-specific Q&A (accuracy: 87%)
- Reduced inference latency by 40% using model distillation

PUBLICATIONS
- "BERT-based Resume Screening" – EMNLP 2024 Workshop
""",

    "David Lee": """
David Lee
Full Stack & Frontend Developer
david.lee@outlook.com

SUMMARY
5 years of experience in web development with strong React and TypeScript skills.
Led frontend teams at two startups.

TECH STACK
Frontend: React, Angular, Vue, HTML, CSS, JavaScript, TypeScript
Backend: Node.js, REST API, GraphQL, FastAPI
Databases: PostgreSQL, MongoDB, SQL
DevOps: Docker, Git, CI/CD
Cloud: AWS
Agile/Scrum

EXPERIENCE
Senior Frontend Engineer – WebAgency (2021–Present)
- Architected React component library used across 5 products
- Led team of 4 engineers; introduced TypeScript codebase migration
- Integrated REST API and GraphQL endpoints

EDUCATION
BE Information Technology – VIT (2020)
""",

    "Eva Chen": """
Eva Chen | Data Analyst
eva.chen@email.com

SKILLS
- Python, SQL
- pandas, matplotlib, seaborn, data analysis
- Power BI, Tableau, data visualization
- Excel, statistics
- Communication, time management, teamwork

EXPERIENCE
Data Analyst – RetailCo (2023–Present)
- Built interactive Power BI dashboards for sales reporting
- Automated data cleaning pipeline in Python (saved 10 hrs/week)
- Wrote complex SQL queries for business intelligence reports

EDUCATION
BBA Business Analytics – Christ University, Bangalore (2023)

Note: No formal ML model building experience but strong analytics background.
""",
}