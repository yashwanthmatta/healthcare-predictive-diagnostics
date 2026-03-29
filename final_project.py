# %%
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_array
import seaborn as sns
import sqlite3
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
import yaml
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Step 1: Dataset Overview
# ---------------------------
file_path = '/Users/yashwanthmatta/Downloads/healthcare_dataset.csv'
data = pd.read_csv(file_path)
print("Dataset loaded successfully!")
print(f"Shape of the dataset: {data.shape}")
print(f"Columns: {list(data.columns)}")
print("Sample Data:\n", data.head())

# ---------------------------
# Step 2: Data Cleaning
# ---------------------------
data = data.drop_duplicates()
data = data[data['Billing Amount'] >= 0]
imputer = SimpleImputer(strategy="mean")
numerical_cols = ['Age', 'Billing Amount']
data[numerical_cols] = imputer.fit_transform(data[numerical_cols])
data = data.dropna()

# ---------------------------
# Step 3: Data Preprocessing
# ---------------------------
categorical_cols = ['Gender', 'Medical Condition', 'Test Results']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

scaler = StandardScaler()
data[['Age', 'Billing Amount']] = scaler.fit_transform(data[['Age', 'Billing Amount']])
data['Hospital Duration'] = (
    pd.to_datetime(data['Discharge Date']) - pd.to_datetime(data['Date of Admission'])
).dt.days
data = data.drop(columns=['Date of Admission', 'Discharge Date'])

# ---------------------------
# Step 4: Data Visualization
# ---------------------------

# Correlation Heatmap
# Ensure only numeric data is used for correlation
numeric_data = data.select_dtypes(include=[np.number])  # Filter numeric columns

# Check if numeric_data is not empty
if not numeric_data.empty:
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()
else:
    print("No numeric columns available for correlation heatmap.")


# Billing Amount Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data['Billing Amount'], kde=True)
plt.title("Distribution of Billing Amount")
plt.xlabel("Billing Amount (Scaled)")
plt.ylabel("Frequency")
plt.show()

# Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data['Age'], kde=True, color='blue')
plt.title("Age Distribution")
plt.xlabel("Age (Scaled)")
plt.ylabel("Frequency")
plt.show()

# Medical Condition Distribution
plt.figure(figsize=(10, 6))
data['Medical Condition'].value_counts().head(10).plot(kind='bar', color='teal')
plt.title("Top 10 Medical Conditions")
plt.xlabel("Medical Condition")
plt.ylabel("Count")
plt.show()

# Gender-wise Billing Amount Distribution
plt.figure(figsize=(8, 5))
sns.boxplot(x='Gender', y='Billing Amount', data=data)
plt.title("Gender-wise Billing Amount Distribution")
plt.xlabel("Gender")
plt.ylabel("Billing Amount (Scaled)")
plt.show()

# Hospital Duration vs Billing Amount
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Hospital Duration', y='Billing Amount', hue='Gender', data=data, palette="Set2")
plt.title("Hospital Duration vs Billing Amount")
plt.xlabel("Hospital Duration (Days)")
plt.ylabel("Billing Amount (Scaled)")
plt.show()

# Hospital Duration Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data['Hospital Duration'], kde=True, color='purple')
plt.title("Distribution of Hospital Duration")
plt.xlabel("Hospital Duration (Days)")
plt.ylabel("Frequency")
plt.show()

# ---------------------------
# Step 5: Data Splitting and Modeling
# ---------------------------
X = data.drop(columns=['Test Results', 'Name', 'Hospital', 'Doctor', 'Room Number'])
y = data['Test Results']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
# Ensure all features in X_train and X_test are numeric
X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

# Train Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)  # Increase max_iter to ensure convergence
logistic_model.fit(X_train, y_train)

# Predict using the test set
y_pred_lr = logistic_model.predict(X_test)

# Calculate accuracy
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr:.2f}")

# Confusion Matrix
# If `label_encoders['Test Results']` exists, retrieve classes; otherwise, set custom labels
if 'Test Results' in label_encoders:
    labels = label_encoders['Test Results'].classes_
else:
    labels = ["Class 0", "Class 1", "Class 2"]  # Adjust as per your dataset's classes

# Plot Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_lr), display_labels=labels).plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Logistic Regression")
plt.show()


# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf), display_labels=label_encoders['Test Results'].classes_).plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Random Forest")
plt.show()

X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

# Define the Stacking Classifier
stacking_model = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),  # Logistic Regression with increased iterations
        ('rf', RandomForestClassifier(random_state=42)),  # Random Forest
        ('knn', KNeighborsClassifier())  # K-Nearest Neighbors
    ],
    final_estimator=LogisticRegression(max_iter=1000)  # Logistic Regression as meta-model
)

# Train the Stacking Classifier
stacking_model.fit(X_train, y_train)

# Make predictions
y_pred_stacking = stacking_model.predict(X_test)

# Evaluate the model
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
print(f"Stacking Classifier Accuracy: {accuracy_stacking:.2f}")

# Plot Confusion Matrix
# Check if label_encoders['Test Results'] exists; use it for display labels, otherwise use generic labels
if 'Test Results' in label_encoders:
    labels = label_encoders['Test Results'].classes_
else:
    labels = ["Class 0", "Class 1", "Class 2"]  # Replace with appropriate class names if known

# Display Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_stacking), display_labels=labels).plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Stacking Classifier")
plt.show()

# ---------------------------
# Step 6: Cross-Validation
# ---------------------------
from sklearn.utils.validation import check_array

# Ensure X and y are numeric and free of NaN or infinite values
try:
    # Convert X to numeric and fill NaN with 0
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    # Convert y to numeric if it's not already
    y = pd.to_numeric(y, errors='coerce').fillna(0)

    # Validate X and y for cross-validation
    check_array(X, ensure_2d=True)  # Ensure X is a valid 2D array
    check_array(y, ensure_2d=False)  # Ensure y is a valid 1D array
    print("X and y are valid for cross-validation.")
except ValueError as e:
    print("Validation Error in X or y:", e)
    raise

# Perform Cross-Validation
try:
    cv_scores = cross_val_score(stacking_model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f}")
except ValueError as e:
    print("Error during cross-validation:", e)
    raise


# ---------------------------
# Step 7: SQL Integration
# ---------------------------
conn = sqlite3.connect("healthcare_analysis.db")
data.to_sql("patient_data", conn, if_exists="replace", index=False)
query = "SELECT Gender, AVG(`Billing Amount`) as Avg_Billing FROM patient_data GROUP BY Gender;"
billing_insights = pd.read_sql_query(query, conn)
print("\nAverage Billing Amount by Gender:")
print(billing_insights)
conn.close()

# ---------------------------




# %%
OpenAI_API_Key = yaml.safe_load(open("/Users/yashwanthmatta/Downloads/credentials.yml"))["openai"]
OpenAI_API_Key

# %%
from openai import OpenAI

model = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key="sk-<your-valid-api-key>")
print("API key is valid.")


# %%
# Step 8: LangChain Integration for AI Summaries
# ---------------------------
try:
    with open("credentials.yml", "r") as f:
        OPENAI_API_KEY = yaml.safe_load(f)["openai"]
except Exception as e:
    raise ValueError("Error loading API key. Ensure 'credentials.yml' exists and is properly configured.") from e

model = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=OPENAI_API_KEY)

prompt_template = ChatPromptTemplate.from_template("""
Generate a summary of the healthcare dataset analysis and model evaluation. Include the following:
- Total number of patients: {total_patients}
- Key medical condition trends: {top_conditions}
- Average hospital duration: {avg_duration}
- Model performance:
    - Logistic Regression Accuracy: {log_accuracy}
    - Random Forest Accuracy: {rf_accuracy}
    - Stacking Classifier Accuracy: {stacking_accuracy}
- SQL Insights:
    - Gender distribution: {gender_insights}
    - Top medical conditions by average billing: {billing_insights}
""")

summary_inputs = {
    "total_patients": len(data),
    "top_conditions": data['Medical Condition'].value_counts().head(3).to_dict(),
    "avg_duration": round(data['Hospital Duration'].mean(), 2),
    "log_accuracy": round(accuracy_lr, 2),
    "rf_accuracy": round(accuracy_rf, 2),
    "stacking_accuracy": round(accuracy_stacking, 2),
    "gender_insights": billing_insights.to_dict('records'),
    "billing_insights": billing_insights.to_dict('records')
}

try:
    response = model.predict(prompt_template.format(**summary_inputs))
    print("\n--- AI Generated Summary ---")
    print(response)
except Exception as e:
    print("Error generating AI summary:", e)

print("\nScript completed successfully!")

# %%



