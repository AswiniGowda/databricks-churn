# Databricks notebook source
# MAGIC %pip install -U databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

dbutils.widgets.text('catalog', '')
dbutils.widgets.text('schema', 'mlops_dev')
catalog = dbutils.widgets.get('catalog')
schema = dbutils.widgets.get('schema')

mlflow.set_registry_uri('databricks-uc')
MODEL = f'{catalog}.{schema}.telco_churn_classifier'
ENDPOINT = f'telco-churn-{schema}'.replace('_', '-')   # e.g. telco-churn-mlops-prod

# 1. Find the Champion version
version = MlflowClient().get_model_version_by_alias(MODEL, 'Champion').version
print(f'Champion is version {version}')

# 2. Create or update the endpoint
w = WorkspaceClient()
entity = ServedEntityInput(
    entity_name=MODEL,
    entity_version=str(version),
    workload_size='Small',
    scale_to_zero_enabled=True,
)

try:
    w.serving_endpoints.get(ENDPOINT)
    exists = True
except Exception:
    exists = False

if exists:
    w.serving_endpoints.update_config_and_wait(name=ENDPOINT, served_entities=[entity])
    print(f'Updated endpoint {ENDPOINT}')
else:
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT,
        config=EndpointCoreConfigInput(name=ENDPOINT, served_entities=[entity]),
    )
    print(f'Created endpoint {ENDPOINT}')

# 3. Test with real rows from the data table
rows = (spark.table(f'{catalog}.{schema}.telco_customers')
        .toPandas().dropna().drop(columns=['customerID', 'Churn']).head(3))
resp = w.serving_endpoints.query(name=ENDPOINT, dataframe_records=rows.to_dict('records'))
print('Predictions:', resp.predictions)