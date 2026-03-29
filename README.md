Healthcare Analytics & Predictive Modeling

End-to-end clinical data pipeline — predictive modeling of patient test results using ensemble ML, SQL-based analytics, and AI-generated insights via LangChain + GPT-4.


Project Overview
Classifying patient test outcomes (Normal / Abnormal / Inconclusive) is a genuinely hard multi-class problem — especially when features like age, billing amount, and medical condition show near-zero correlation with the target. This project tackles that challenge head-on using an ensemble stacking architecture and augments the analysis with SQL-driven operational insights and AI-powered narrative summaries.
Built for: BANA 6620 Computing for Business Analytics | University of Colorado Denver
Authors: Yashwanth Goud Matta 

Results at a Glance
ModelAccuracyNotesLogistic Regression33%Baseline — struggles with near-uniform class distributionRandom Forest42%Captures non-linear relationships; still affected by class balanceStacking Classifier (LR + RF + KNN)42%Best overall stability across all three classesCross-Validation (5-fold, Stacking)40%Robust estimate across multiple data splits
Why are accuracies modest? The correlation heatmap reveals near-zero relationships between all features and Test Results. This is a genuinely difficult prediction problem — the dataset's synthetic nature means test outcomes are largely independent of demographics and billing. The value of this project lies in the pipeline architecture, the SQL operational layer, and the AI summary integration — not in squeezing accuracy on a near-random target.

What I Built
1. Data Preprocessing Pipeline

Loaded 55,500-row healthcare dataset (15 columns) from Kaggle
Removed 534 duplicate records and filtered invalid billing entries
Applied SimpleImputer (mean strategy) for missing values in Age and Billing Amount
LabelEncoder for categorical features: Gender, Medical Condition, Test Results
StandardScaler for Age and Billing Amount normalization
Engineered Hospital Duration feature from admission/discharge date difference

2. Exploratory Data Analysis (EDA)

Pairplot of key features: Age, Billing Amount, Hospital Duration, Test Results
Correlation heatmap — revealed near-zero correlation between features and target (key finding)
Billing amount distribution — near-uniform distribution across $10K–$30K range
Hospital duration distribution — relatively uniform across 0–30 days
Gender-wise billing analysis — no significant gender-based billing disparity

3. Predictive Modeling — Three Architectures
Logistic Regression (Baseline)

33% accuracy — establishes the difficulty floor for this classification task
Useful for understanding linear feature-outcome relationships

Random Forest Classifier

42% accuracy — captures non-linear patterns missed by logistic regression
Better balanced precision/recall across Abnormal, Inconclusive, and Normal classes

Stacking Classifier (Best Model)

Base estimators: Logistic Regression + Random Forest + K-Nearest Neighbors
Meta-estimator: Logistic Regression
42% accuracy with improved prediction stability vs. single models
5-fold cross-validation confirms robust generalization at 40% mean accuracy

4. SQL Integration — Operational Analytics
Saved preprocessed data to SQLite database and ran 3 analytical queries:
sql-- Patient distribution by gender
SELECT Gender, COUNT(*) AS Patient_Count 
FROM patient_data GROUP BY Gender;
-- Result: 27,411 male | 27,449 female (near-perfect parity)

-- Average billing by medical condition
SELECT Medical_Condition, AVG(Billing_Amount) AS Avg_Billing
FROM patient_data GROUP BY Medical_Condition ORDER BY Avg_Billing DESC;

-- High-risk long-stay patients
SELECT Name, Medical_Condition, Hospital_Duration
FROM patient_data WHERE Hospital_Duration > 10;
-- Result: 36,559 patients with stays > 10 days
5. LangChain + GPT-4 AI Summaries
Integrated OpenAI's GPT-4 via LangChain to auto-generate natural language summaries of:

Patient population trends
Model performance interpretation
SQL insight narrative
Actionable clinical and operational recommendations


Key Findings

Near-perfect gender parity: 27,411 male vs. 27,449 female patients — no gender bias in treatment volume
Top medical conditions: Diabetes, Hypertension, and Asthma are the three most prevalent chronic conditions
Uniform billing: Average billing ~$23,452 with consistent distribution across demographics — suggesting standardized cost practices
Long-stay burden: 36,559 of 54,860 patients (66.6%) stayed longer than 10 days — significant resource planning implication
Class imbalance challenge: All three test result classes (Normal/Abnormal/Inconclusive) are nearly equally distributed, making classification inherently difficult without feature engineering


Dataset
Source: Healthcare Dataset — Kaggle
PropertyDetailRows55,500 patient recordsColumns15 featuresTargetTest Results (Normal / Abnormal / Inconclusive)Key FeaturesAge, Gender, Medical Condition, Billing Amount, Hospital Duration, Admission TypeData QualityDuplicates removed (54,966 → 54,860 after cleaning)

Tech Stack
CategoryToolsLanguagePython 3ML & ModelingScikit-learn (LogisticRegression, RandomForestClassifier, StackingClassifier, KNeighborsClassifier)Data ProcessingPandas, NumPy, SimpleImputer, StandardScaler, LabelEncoderVisualizationMatplotlib, Seaborn (pairplot, heatmap, histplot, boxplot)DatabaseSQLite3 — data storage and SQL query analyticsAI IntegrationLangChain, OpenAI GPT-4 — automated insight generationValidationcross_val_score (5-fold CV)

Project Structure
healthcare-predictive-diagnostics/
│
├── final_project.py            # Complete Python pipeline — all steps end to end
├── README.md
└── data/
    └── healthcare_dataset.csv  # Download from Kaggle link above

How to Run
1. Clone the repo
bashgit clone https://github.com/yashwanthmatta/healthcare-predictive-diagnostics.git
cd healthcare-predictive-diagnostics
2. Install dependencies
bashpip install pandas numpy matplotlib seaborn scikit-learn langchain openai pyyaml
3. Download the dataset
Get healthcare_dataset.csv from Kaggle and place it in /data.
4. Configure OpenAI API (for AI summary section)
Create credentials.yml in the project root:
yamlopenai: your-openai-api-key-here
5. Run the full pipeline
bashpython final_project.py
The script runs all steps in sequence: data loading → cleaning → EDA → modeling → SQL analytics → AI summary.

What I Learned
This project taught me something more valuable than a high accuracy score: how to honestly interpret a hard classification problem. When the correlation heatmap showed near-zero relationships between all features and the target, I didn't try to force the numbers up artificially. Instead, I built a robust ensemble architecture, validated it properly with cross-validation, and used SQL and AI tooling to extract operational value that a pure ML approach would miss.
That combination — rigorous modeling, honest evaluation, and multi-layer analytics — is what real-world healthcare data science looks like.
