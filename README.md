Customer Churn Prediction
Project Description
This project aims to develop a predictive model to identify customers who are likely to churn. The dataset used for this project is the Telco Customer Churn dataset. Various machine learning algorithms are applied to predict churn, and strategies are proposed to improve customer retention.

Dataset
Dataset Used: Telco Customer Churn
Source: Telco Customer Churn Dataset
Skills Demonstrated
Data Cleaning and Preparation
Statistical Analysis
Predictive Modeling
Machine Learning (Logistic Regression, Random Forest)
Data Visualization (PowerBI)
Steps
Data Cleaning and Preparation:

Load the dataset and inspect for missing values.
Handle missing values and clean the data.
Convert categorical variables into numerical formats using encoding.
Statistical Analysis:

Perform exploratory data analysis (EDA) to understand the data distribution and relationships.
Visualize the data using histograms, box plots, and correlation matrices.
Predictive Modeling:

Split the dataset into training and testing sets.
Train machine learning models (Logistic Regression and Random Forest) to predict customer churn.
Evaluate the models using accuracy, precision, recall, F1 score, and ROC AUC.
Data Visualization with PowerBI:

Import the cleaned data into PowerBI.
Create visualizations to showcase:
Churn distribution
Feature importance (based on Random Forest)
Model performance metrics (Accuracy, Precision, Recall, F1 Score)
Customer segmentation and retention strategies
Installation and Setup
Clone the Repository:

bash
Copy code
git clone https://github.com/tanmay003MetricsMaestro/customer-churn-prediction.git
cd customer-churn-prediction
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Python Script:

bash
Copy code
python churn_prediction.py
Open PowerBI:

Import cleaned_telco_data.csv into PowerBI to create visualizations.
Python Script Outline
python
Copy code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load Data
df = pd.read_csv('Telco-Customer-Churn.csv')

# Data Cleaning and Preparation
# (Handle missing values, encode categorical variables, etc.)

# Split Data
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Models
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

# Predict and Evaluate
y_pred_log_reg = log_reg.predict(X_test)
y_pred_rf = rf_clf.predict(X_test)

# Evaluation Metrics
metrics = {
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [accuracy_score(y_test, y_pred_log_reg), accuracy_score(y_test, y_pred_rf)],
    'Precision': [precision_score(y_test, y_pred_log_reg), precision_score(y_test, y_pred_rf)],
    'Recall': [recall_score(y_test, y_pred_log_reg), recall_score(y_test, y_pred_rf)],
    'F1 Score': [f1_score(y_test, y_pred_log_reg), f1_score(y_test, y_pred_rf)],
    'ROC AUC': [roc_auc_score(y_test, y_pred_log_reg), roc_auc_score(y_test, y_pred_rf)]
}

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('model_performance.csv', index=False)

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_clf.feature_importances_
})
feature_importance.to_csv('feature_importance.csv', index=False)

# Save Cleaned Data
df.to_csv('cleaned_telco_data.csv', index=False)
PowerBI Visualization Steps
Import Data into PowerBI:

Open PowerBI Desktop.
Click on "Get Data" and select "Text/CSV".
Load cleaned_telco_data.csv.
Create Visualizations:

Churn Distribution:
Add a new Pie Chart or Bar Chart.
Drag Churn to the Values field.
Feature Importance:
Import feature_importance.csv.
Create a Bar Chart with Feature and Importance.
Model Performance Metrics:
Import model_performance.csv.
Create a Table or Column Chart to display metrics.
Customer Segmentation and Retention Strategies:
Use features like tenure, TotalCharges, TotalServices.
Create scatter plots, clustered bar charts, or treemaps for segmentation.
Contributors
Your Name - tanmay003MetricsMaestro
License
This project is licensed under the MIT License - see the LICENSE.md file for details.

