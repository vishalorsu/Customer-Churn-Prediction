# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 03:42:19 2023

@author: Vishal Orsu
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/Vishal Orsu/OneDrive/Desktop/Ait664-Final Project/credit_card_churn.csv/credit_card_churn.csv")


columns_to_drop = [
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
]

data = data.drop(columns=columns_to_drop)

# Checking for missing values
print(data.isnull().sum())

# Removing duplicates
duplicates = data.duplicated(keep=False)
print(data[duplicates])
data = data.drop_duplicates()
duplicates = data.duplicated(keep=False)
print(data[duplicates])


print(data['Customer_Age'].skew())

# Checking for outliers
#numeric_cols = ['Customer_Age', 'Months_on_book', 'Total_Relationship_Count', 'Total_Trans_Amt', 'Total_Trans_Ct']
#for col in numeric_cols:
   # data.boxplot(column=col)


numeric_cols = ['Customer_Age', 'Months_on_book', 'Total_Relationship_Count', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
for col in numeric_cols:
    data.boxplot(column=col)


# Describing a specific column (replacing 'column_name' with the actual column name)
print(data['Customer_Age'].describe())

# Calculating and printing attribute correlation
correlation_matrix = data[numeric_cols].corr()
print("Attribute Correlation:")
print(correlation_matrix)



# Loading the cleaned dataset
data = pd.read_csv("C:/Users/Vishal Orsu/OneDrive/Desktop/Ait664-Final Project/credit_card_churn.csv/credit_card_churn.csv")


#  heatmap to visualize the correlation between attributes
corr_matrix = data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Attributes")
plt.show()


##########################################################################################################################################################################
#Univariate 

import seaborn as sns
import matplotlib.pyplot as plt

# Education Level Count Plot
plt.figure(figsize=(10, 6))
sns.countplot(data['Education_Level'], order=data['Education_Level'].value_counts().index, palette='viridis')
plt.title('Plot of Education Level')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

# Customer Age Histogram
plt.figure(figsize=(10, 6))
data['Customer_Age'].plot(kind='hist', edgecolor='black', color='lightgreen')
plt.title('Histogram for Customer Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

###########################################################################################################################################################################
#Bivariate 
plt.scatter(data['Customer_Age'], data['Total_Trans_Amt'], alpha=0.5)
plt.title('Customer Age vs Total Transaction Amount')
plt.xlabel('Customer Age')
plt.ylabel('Total Transaction Amount')
plt.show()


# bar plot of income category vs. total transaction amount
sns.barplot(x='Income_Category', y='Total_Trans_Amt', data=data, ci=None, palette='Set2')
plt.title('Income Category vs. Total Transaction Amount')
plt.xlabel('Income Category')
plt.ylabel('Total Transaction Amount')
plt.xticks(rotation=45)
plt.show()



# bar plot of income category vs. total transaction amount
sns.barplot(x='Attrition_Flag', y='Contacts_Count_12_mon', data=data, ci=None, palette='Set2')
plt.title('Attrition Flag vs. Contacts_Count_12_mon' )
plt.xlabel('Attrition Flag')
plt.ylabel('Contacts_Count_12_mon')
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(x="Attrition_Flag", y="Contacts_Count_12_mon", data=data)
plt.title("Attrition Flag vs Contacts Count (12 months)")
plt.xlabel("Attrition Flag")
plt.ylabel("Contacts Count (12 months)")
plt.show()
###############################################################

# Bivariate

# Scatter plot of Customer Age vs. Total Transaction Amount
plt.figure(figsize=(10, 6))
plt.scatter(data['Customer_Age'], data['Total_Trans_Amt'], alpha=0.5, color='orange')
plt.title('Customer Age vs Total Transaction Amount')
plt.xlabel('Customer Age')
plt.ylabel('Total Transaction Amount')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Line plot of Customer Age vs. Total Transaction Amount
plt.figure(figsize=(10, 6))
sns.lineplot(x='Customer_Age', y='Total_Trans_Amt', data=data, ci=None, color='blue')

plt.title('Line Plot: Customer Age vs Total Transaction Amount')
plt.xlabel('Customer Age')
plt.ylabel('Total Transaction Amount')
plt.show()




import seaborn as sns
import matplotlib.pyplot as plt

# Bar plot of Income Category vs Total Transaction Amount
plt.figure(figsize=(12, 6))
sns.barplot(x='Income_Category', y='Total_Trans_Amt', data=data, ci=None, palette='pastel')
plt.title('Income Category vs Total Transaction Amount')
plt.xlabel('Income Category')
plt.ylabel('Total Transaction Amount')
plt.xticks(rotation=45)
plt.ylim(0, 5000) 
plt.show()


############################
#REASEARCH QUESTION 1:
    
# Converting Attrition_Flag to numeric values
data['Attrition_Flag'] = data['Attrition_Flag'].map({'Attrited Customer': 1, 'Existing Customer': 0})

# bar plot for Attrition Rate by Gender

#gender_attrition = data.groupby('Gender')['Attrition_Flag'].mean()
#gender_attrition.plot(kind='bar', title="Attrition Rate by Gender", ylabel="Attrition Rate", grid=False)
#plt.show()


plt.style.use('seaborn-whitegrid')

# Attrition Rate by Gender bar plot
gender_attrition = data.groupby('Gender')['Attrition_Flag'].mean()
gender_attrition.plot(kind='bar', title="Attrition Rate by Gender", ylabel="Attrition Rate", color=['skyblue', 'lightcoral'], edgecolor='k')


plt.legend().set_visible(False)
for i, value in enumerate(gender_attrition):
    plt.text(i, value, f"{value:.2%}", ha='center', va='bottom')


plt.xlabel("Gender")
plt.tight_layout()
plt.show()

print("Attrition Rate by Gender Data:")
print(gender_attrition)

print(data['Gender'].describe())


#  Attrition Rate by Education Level
#education_attrition = data.groupby('Education_Level')['Attrition_Flag'].mean()
#education_attrition.plot(kind='bar', title="Attrition Rate by Education Level", ylabel="Attrition Rate", grid=False)
#plt.show()

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

#  Attrition Rate by Education Level Bar plot 
education_attrition = data.groupby('Education_Level')['Attrition_Flag'].mean()
ax = education_attrition.plot(kind='bar', title="Attrition Rate by Education Level", ylabel="Attrition Rate", color='skyblue', edgecolor='k')
ax.get_legend().remove()
for i, value in enumerate(education_attrition):
    plt.text(i, value, f"{value:.2%}", ha='center', va='bottom')

plt.xticks(rotation=45)
plt.xlabel("Education Level")
plt.ylabel("Attrition Rate")
plt.tight_layout()
plt.show()

print("Attrition Rate by Education Level Data:")
print(education_attrition)
print(data['Education_Level'].describe())


#  Attrition Rate by Marital Status
marital_attrition = data.groupby('Marital_Status')['Attrition_Flag'].mean()
marital_attrition.plot(kind='bar', title="Attrition Rate by Marital Status", ylabel="Attrition Rate", grid=False)
plt.show()

#  Attrition Rate by Income Category
income_attrition = data.groupby('Income_Category')['Attrition_Flag'].mean()
income_attrition.plot(kind='bar', title="Attrition Rate by Income Category", ylabel="Attrition Rate", grid=False)
plt.show()



import matplotlib.pyplot as plt

# Attrition Rate by Marital Status
marital_attrition = data.groupby('Marital_Status')['Attrition_Flag'].mean()
marital_attrition.plot(kind='bar', color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black', figsize=(8, 6))

plt.title("Attrition Rate by Marital Status", fontsize=16)
plt.xlabel("Marital Status", fontsize=14)
plt.ylabel("Attrition Rate", fontsize=14)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# Attrition Rate by Income Category
income_attrition = data.groupby('Income_Category')['Attrition_Flag'].mean()
income_attrition.plot(kind='bar', color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#34495e'], edgecolor='black', figsize=(8, 6))

plt.title("Attrition Rate by Income Category", fontsize=16)
plt.xlabel("Income Category", fontsize=14)
plt.ylabel("Attrition Rate", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

################################
#data_summary = data.describe(include='all')
#print(data_summary)
########################

#RESEARCH QUESTION 2:
    
# Filtering data for customers who experienced attrition (1) and those who didnt (0)
attrited_customers = data[data['Attrition_Flag'] == 1]
existing_customers = data[data['Attrition_Flag'] == 0]

# Analyzing transaction patterns for Total_Trans_Amt using distribution plots
plt.figure(figsize=(10, 6))
sns.kdeplot(attrited_customers['Total_Trans_Amt'], label='Attrited Customers', color='red', shade=True)
sns.kdeplot(existing_customers['Total_Trans_Amt'], label='Existing Customers', color='blue', shade=True)
plt.title('Total Transaction Amount Distribution')
plt.xlabel('Total Transaction Amount')
plt.ylabel('Density')
plt.legend()
plt.show()

# Analyzing transaction patterns for Total_Trans_Ct using distribution plots
plt.figure(figsize=(10, 6))
sns.kdeplot(attrited_customers['Total_Trans_Ct'], label='Attrited Customers', color='red', shade=True)
sns.kdeplot(existing_customers['Total_Trans_Ct'], label='Existing Customers', color='blue', shade=True)
plt.title('Total Transaction Count Distribution')
plt.xlabel('Total Transaction Count')
plt.ylabel('Density')
plt.legend()
plt.show()

#########################################################################################################
#LOGISTIC REGRESSION 
#RQ1 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

df = pd.read_csv('/Users/rishikreddy/Desktop/credit_card_churn_cleaned.csv')

# Data preprocessing
# Encoding categorical variables
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Education_Level'] = le.fit_transform(df['Education_Level'])
df['Marital_Status'] = le.fit_transform(df['Marital_Status'])
df['Income_Category'] = le.fit_transform(df['Income_Category'])
df['Card_Category'] = le.fit_transform(df['Card_Category'])

# Feature selection
features = ['Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status',
            'Income_Category', 'Card_Category', 'Months_on_book', 'Total_Relationship_Count',
            'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

# Logistic Regression 
X = df[features]
y = df['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)

# training and testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model with increased max_iter
logreg_model = LogisticRegression(max_iter=1000) 
logreg_model.fit(X_train, y_train)

# Predict on the test set
y_pred = logreg_model.predict(X_test)

# Calculating accuracy, probability scores for the positive class
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy * 100:.2f}%')

from sklearn.metrics import roc_curve, auc


y_prob = logreg_model.predict_proba(X_test)[:, 1]

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

print('ROCAUC:')
print(roc_auc)

# printing feature coefficients
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': logreg_model.coef_[0]
})

# Sorting features by absolute coefficient values
coefficients['Absolute Coefficient'] = coefficients['Coefficient'].abs()
coefficients = coefficients.sort_values(by='Absolute Coefficient', ascending=False)

# top features influencing credit card attrition
print(coefficients[['Feature', 'Coefficient']])


# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
from sklearn.metrics import classification_report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Confusion Matrix Plot 
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

##################################################################################################
#LOGISTIC REGRESSION 
#XGBOOST 
#RQ1 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

#df = pd.read_csv('/Users/rishikreddy/Desktop/credit_card_churn_cleaned.csv')

# Data preprocessing
# Encoding categorical variables
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Education_Level'] = le.fit_transform(df['Education_Level'])
df['Marital_Status'] = le.fit_transform(df['Marital_Status'])
df['Income_Category'] = le.fit_transform(df['Income_Category'])
df['Card_Category'] = le.fit_transform(df['Card_Category'])

# Feature selection
features = ['Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status',
            'Income_Category', 'Card_Category', 'Months_on_book', 'Total_Relationship_Count',
            'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

# Logistic Regression for Credit Card Attrition
X = df[features]
y = df['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)

# training and testing sets (Split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the log regression model with increased max_iter
logreg_model = LogisticRegression(max_iter=1000) 
logreg_model.fit(X_train, y_train)

# Predicting on the test set
y_pred_logreg = logreg_model.predict(X_test)

# accuracy
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f'Logistic Regression Accuracy: {accuracy_logreg * 100:.2f}%')

# ROC curve and AUC for Logistic Regression
y_prob_logreg = logreg_model.predict_proba(X_test)[:, 1]
fpr_logreg, tpr_logreg, thresholds_logreg = roc_curve(y_test, y_prob_logreg)
roc_auc_logreg = auc(fpr_logreg, tpr_logreg)

# ROC curve for Logistic Regression
plt.figure(figsize=(8, 6))
plt.plot(fpr_logreg, tpr_logreg, color='darkorange', lw=2, label=f'LogReg ROC curve (area = {roc_auc_logreg:.2f})')

# XGBoost for Credit Card Attrition
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# Predicting on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Accuracy for XGBoost
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f'XGBoost Accuracy: {accuracy_xgb * 100:.2f}%')

# ROC curve and AUC for XGBoost
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_prob_xgb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# plot
plt.plot(fpr_xgb, tpr_xgb, color='green', lw=2, label=f'XGBoost ROC curve (area = {roc_auc_xgb:.2f})')


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

print('Logistic Regression ROCAUC:', roc_auc_logreg)
print('XGBoost ROCAUC:', roc_auc_xgb)

# printing the feature coefficients for Logistic Regression
coefficients_logreg = pd.DataFrame({
    'Feature': features,
    'Coefficient': logreg_model.coef_[0]
})

# Sorting the features by absolute coefficient values
coefficients_logreg['Absolute Coefficient'] = coefficients_logreg['Coefficient'].abs()
coefficients_logreg = coefficients_logreg.sort_values(by='Absolute Coefficient', ascending=False)

# Displaying the top features influencing credit card attrition for Logistic Regression
print('Logistic Regression Coefficients:')
print(coefficients_logreg[['Feature', 'Coefficient']])

# Confusion Matrix for Logistic Regression
conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)
print("Confusion Matrix for Logistic Regression:")
print(conf_matrix_logreg)

# Classification Report for Logistic Regression
class_report_logreg = classification_report(y_test, y_pred_logreg)
print("Classification Report for Logistic Regression:")
print(class_report_logreg)

# Confusion Matrix Plot for Logistic Regression
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_logreg, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Confusion Matrix for XGBoost
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
print("Confusion Matrix for XGBoost:")
print(conf_matrix_xgb)

# Classification Report for XGBoost
class_report_xgb = classification_report(y_test, y_pred_xgb)
print("Classification Report for XGBoost:")
print(class_report_xgb)

# Confusion Matrix Plot for XGBoost
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix for XGBoost')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

#######
# feature coefficients for Logistic Regression
coefficients_logreg = pd.DataFrame({
    'Feature': features,
    'Coefficient': logreg_model.coef_[0]
})

# Sorting the feature by absolute coefficient values
coefficients_logreg['Absolute Coefficient'] = coefficients_logreg['Coefficient'].abs()
coefficients_logreg = coefficients_logreg.sort_values(by='Absolute Coefficient', ascending=False)

# Showing the top features influencing credit card attrition for Logistic Regression
print('Logistic Regression Coefficients:')
print(coefficients_logreg[['Feature', 'Coefficient']])


# Segmenting based on a key feature (e.g Contacts_Count_12_mon)
attrited_customers = df[df['Attrition_Flag'] == 'Attrited Customer']
non_attrited_customers = df[df['Attrition_Flag'] == 'Existing Customer']


print(attrited_customers[['Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status',
            'Income_Category', 'Card_Category', 'Months_on_book', 'Total_Relationship_Count',
            'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']])
print(non_attrited_customers[['Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status',
            'Income_Category', 'Card_Category', 'Months_on_book', 'Total_Relationship_Count',
            'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']])


# Box plot for key feature
sns.boxplot(x='Attrition_Flag', y= 'Months_Inactive_12_mon' , data=df)
plt.title('Distribution by Attrition')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt


sns.set(style='whitegrid')

# Box plot for key feature
plt.figure(figsize=(10, 6))
sns.boxplot(x='Attrition_Flag', y='Months_Inactive_12_mon', data=df, palette='Set3')
plt.title('Distribution by Attrition', fontsize=16)
plt.xlabel('Attrition Flag', fontsize=14)
plt.ylabel('Months Inactive (12 months)', fontsize=14)
plt.show()


##################################################################################################
#RANDOM FOREST 
# Transactions over time line plot 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap


credit_card_data = pd.read_csv('/Users/rishikreddy/Desktop/credit_card_churn_cleaned.csv')

credit_card_data['Months_on_book'] = pd.to_datetime(credit_card_data['Months_on_book'])

credit_card_data = credit_card_data.sort_values('Months_on_book')

plt.figure(figsize=(12, 6))
sns.lineplot(x='Months_on_book', y='Total_Trans_Amt', data=credit_card_data, hue='Attrition_Flag')
plt.title('Transaction Patterns Over Time')
plt.xlabel('Months on Book')
plt.ylabel('Total Transaction Amount')
plt.show()

# Feature engineering and preprocessing
credit_card_data = pd.get_dummies(credit_card_data, columns=['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'])

# training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    credit_card_data.drop('Attrition_Flag', axis=1),
    credit_card_data['Attrition_Flag'],
    test_size=0.2,
    random_state=42
)

# Dropping the datetime column before training the model
X_train = X_train.drop('Months_on_book', axis=1)
X_test = X_test.drop('Months_on_book', axis=1)

# Train
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model evaluation
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type='bar')


# Plot for Feature Importance 
feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
feature_importance = feature_importance.sort_values(ascending=False)

# Plotting the top N features contributing to the models predictions
top_n = 10  # Choosing the number of top features to display
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance.head(top_n), y=feature_importance.head(top_n).index, palette="viridis")
plt.title(f'Top {top_n} Features Contributing to Model Predictions')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.show()


# Plot for Model Misclassification 
conf_matrix = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap="Blues")
plt.title('Confusion Matrix - Model Performance')
plt.show()



#########################################################################################3
#RANDOM FOREST FEATURE IMPORTANCE (EXTRA, NOT USED IN THE FINAL)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
 

df = pd.read_csv('/Users/rishikreddy/Desktop/credit_card_churn_cleaned.csv') 

# Separate categorical and numerical columns
categorical_features = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
numerical_features = [col for col in df.columns if col not in categorical_features]

# Encode categorical variables
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_features = encoder.fit_transform(df[categorical_features])
encoded_df = pd.concat([df[numerical_features], pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))], axis=1)

# Define target variable and features
target_variable = 'Attrition_Flag'
features = encoded_df.columns.tolist()
features.remove(target_variable)

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(encoded_df[features], df[target_variable], test_size=0.2, random_state=42)

# Train a machine learning model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Plot feature importance
importances = model.feature_importances_
indices = range(len(importances))

plt.figure(figsize=(10, 6))
plt.barh(indices, importances, align='center')
plt.yticks(indices, features)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest Classifier - Feature Importance')
plt.show()

#####################################################

####################################

#BOX PLOTS RQ2

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

df = pd.read_csv('/Users/rishikreddy/Desktop/credit_card_churn_cleaned.csv')

# Grouping by Attrition_Flag
attrition_grouped = df.groupby('Attrition_Flag')

# Transaction Feature exploration
transaction_features = ['Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1']
print("Descriptive Statistics for Transaction Features:")
print(attrition_grouped[transaction_features].describe())

bright_palette = sns.color_palette("bright")
# Visualizing the Data
for feature in transaction_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Attrition_Flag', y=feature, data=df, palette=bright_palette)
    plt.title(f'Boxplot of {feature} by Attrition_Flag')
    plt.show()

# Correlation Analysis
correlation_matrix = df[transaction_features + ['Attrition_Flag']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Transaction Features and Attrition_Flag')
plt.show()

# Testing
print("T-test Results for Transaction Features:")
for feature in transaction_features:
    t_test_result = ttest_ind(df[df["Attrition_Flag"] == "Existing Customer"][feature],
                               df[df["Attrition_Flag"] == "Attrited Customer"][feature])
    print(f'T-test for {feature}:')
    print(f'  - T-statistic: {t_test_result.statistic}')
    print(f'  - P-value: {t_test_result.pvalue}')
    print('-' * 30)
    
###########################################################################