# Databricks notebook source
import pandas as pd
import mlflow
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

dbutils.widgets.text('catalog', '')
dbutils.widgets.text('schema', 'mlops_dev')
catalog = dbutils.widgets.get('catalog')
schema = dbutils.widgets.get('schema')

spark.sql(f'CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}')
mlflow.set_registry_uri('databricks-uc')
MODEL = f'{catalog}.{schema}.telco_churn_classifier'

# 1. Load and clean the raw data
URL = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
df = pd.read_csv(URL)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # blank strings become NaN
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

# 2. Save the clean table so batch inference reads the same data
spark.createDataFrame(df).write.mode('overwrite').saveAsTable(f'{catalog}.{schema}.telco_customers')

# 3. Split features and target
y = df['Churn']
X = df.drop(columns=['customerID', 'Churn'])
num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [c for c in X.columns if c not in num_cols]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pre = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
])

# 4. Train and log
with mlflow.start_run():
    model = Pipeline([
        ('pre', pre),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ]).fit(Xtr, ytr)
    pred = model.predict(Xte)
    mlflow.log_param('model', 'logistic_regression_balanced')
    mlflow.log_metric('accuracy', accuracy_score(yte, pred))
    mlflow.log_metric('recall', recall_score(yte, pred))
    mlflow.log_metric('f1', f1_score(yte, pred))
    mlflow.log_metric('roc_auc', roc_auc_score(yte, model.predict_proba(Xte)[:, 1]))
    info = mlflow.sklearn.log_model(
        model,
        name='model',  # use artifact_path= on older MLflow versions
        signature=infer_signature(Xtr, model.predict(Xtr)),
        registered_model_name=MODEL,
    )

dbutils.jobs.taskValues.set(key='model_version', value=int(info.registered_model_version))