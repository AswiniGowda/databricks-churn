# Databricks notebook source
import mlflow

dbutils.widgets.text('catalog', '')
dbutils.widgets.text('schema', 'mlops_dev')
catalog = dbutils.widgets.get('catalog')
schema = dbutils.widgets.get('schema')

mlflow.set_registry_uri('databricks-uc')
MODEL = f'{catalog}.{schema}.telco_churn_classifier'

model = mlflow.pyfunc.load_model(f'models:/{MODEL}@Champion')

customers = spark.table(f'{catalog}.{schema}.telco_customers').toPandas()
features = customers.drop(columns=['customerID', 'Churn'])

out = customers[['customerID', 'Churn']].rename(columns={'Churn': 'actual_churn'})
out['predicted_churn'] = model.predict(features)

spark.createDataFrame(out).write.mode('overwrite').saveAsTable(f'{catalog}.{schema}.telco_churn_predictions')