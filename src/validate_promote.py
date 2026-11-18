# Databricks notebook source
import mlflow
from mlflow import MlflowClient

dbutils.widgets.text('catalog', '')
dbutils.widgets.text('schema', 'mlops_dev')
dbutils.widgets.text('min_f1', '0.55')
catalog = dbutils.widgets.get('catalog')
schema = dbutils.widgets.get('schema')
min_f1 = float(dbutils.widgets.get('min_f1'))

mlflow.set_registry_uri('databricks-uc')
client = MlflowClient()
MODEL = f'{catalog}.{schema}.telco_churn_classifier'

version = dbutils.jobs.taskValues.get(taskKey='train', key='model_version', debugValue=1)
mv = client.get_model_version(MODEL, str(version))
f1 = mlflow.get_run(mv.run_id).data.metrics['f1']

try:
    champ = client.get_model_version_by_alias(MODEL, 'Champion')
    champ_f1 = mlflow.get_run(champ.run_id).data.metrics['f1']
except Exception:
    champ_f1 = 0.0  # no Champion yet

if f1 >= min_f1 and f1 >= champ_f1:
    client.set_registered_model_alias(MODEL, 'Champion', str(version))
    print(f'Promoted version {version} to Champion (f1 {f1:.3f})')
else:
    raise Exception(f'Gate failed: f1 {f1:.3f}, minimum {min_f1}, current Champion {champ_f1:.3f}')