# Telco Customer Churn: End-to-End MLOps on Azure Databricks

A small, low-cost, fully automated MLOps pipeline. Nobody opens a notebook: a merge to `main` trains a model, checks its quality, promotes it only if it passes, scores every customer in batch, and deploys a real-time serving endpoint.

```
git push -> GitHub Actions -> databricks bundle deploy -> serverless job:

                         train -> validate_promote -+-> batch_inference
                                                    +-> deploy_endpoint
```

**Stack:** Azure Databricks (serverless), Databricks Asset Bundles, MLflow, Unity Catalog, Delta tables, Model Serving, GitHub Actions.

---

## 1. What this project does

| Task | What it does |
|---|---|
| `train` | Downloads the IBM Telco Customer Churn CSV, cleans it, saves it as a Delta table, trains a logistic regression pipeline, logs metrics to MLflow, and registers the model in Unity Catalog. |
| `validate_promote` | Quality gate. The new model becomes the `Champion` alias only if its F1 score is at least `min_f1` and at least the current Champion's F1. Otherwise the job fails and the Champion stays unchanged. |
| `batch_inference` | Loads the model by the `Champion` alias, scores every customer, and writes predictions to a Delta table. |
| `deploy_endpoint` | Creates or updates a real-time Model Serving endpoint that serves the current `Champion` version, waits until it is ready, and sends a test request. |

**Why F1 and not accuracy:** only about a quarter of customers churn, so a model that always predicts "no churn" would look about 73% accurate. F1 punishes that.

**Example result from a real run:** F1 0.614, recall 0.783, accuracy 0.738, ROC AUC 0.841. Your numbers may differ slightly by library version.

**Dataset:** the IBM Telco Customer Churn sample dataset (7,043 rows, 21 columns). It is a public sample dataset about a fictional telecom company.

**Two ways to get predictions:**
- **Batch:** the `telco_churn_predictions` table holds a score for every customer. No endpoint, no URL, no idle cost.
- **Real-time:** the serving endpoint returns one prediction per request. It is only worth it if an application needs an answer immediately.

---

## 2. Repository layout

```
mlops-telco-churn/
├─ databricks.yaml                 # bundle settings: variables, targets
├─ resources/
│  └─ training_job.yml             # the job: 4 tasks, order, parameters, paused schedule
├─ src/
│  ├─ train.py                     # task 1
│  ├─ validate_promote.py          # task 2 (quality gate)
│  ├─ batch_inference.py           # task 3
│  └─ endpoint.py                  # task 4 (real-time endpoint)
├─ .github/workflows/
│  └─ deploy.yaml                  # CI/CD: validate on PR, deploy and run on merge
├─ test_endpoint.py                # optional client to call the endpoint
├─ .gitignore
└─ README.md
```

**Important file rules**
- `databricks.yaml` includes `resources/*.yml`. The job file must end in `.yml`, not `.yaml`, or the bundle silently finds no job.
- Never put Python code in a `.yml` file inside `resources/`. Everything in that folder is read as bundle configuration.
- `endpoint.py` is a Python file in `src/`, not a YAML file.

---

## 3. How it works (technical overview)

1. `databricks bundle deploy` uploads `src/*.py` into the workspace and creates a **Job** from the YAML files. The first line of each Python file, `# Databricks notebook source`, makes Databricks treat it as a notebook.
2. `databricks bundle run` calls the Jobs API to start the job. Databricks runs the tasks in dependency order, each on its own serverless compute. The run is labelled **Manually** because it was started by an API call, not by the schedule.
3. Parameters from `base_parameters` reach the notebooks as widgets (`dbutils.widgets.get(...)`).
4. `train` passes the model version number to the next task with `dbutils.jobs.taskValues`.
5. `validate_promote` reads the metrics from MLflow and sets the `Champion` alias, or raises an exception to fail the job.
6. `batch_inference` loads `models:/<catalog>.<schema>.telco_churn_classifier@Champion`, so it never hard-codes a version.
7. `deploy_endpoint` looks up the `Champion` version and points a serving endpoint at it.

**Targets:** the bundle defines `dev` and `prod`. Both can point at the same workspace and differ by schema (`mlops_dev`, `mlops_prod`) and deployment path. In a real company they would be separate workspaces. You can keep just one target if you only use one. The GitHub workflow uses `-t prod`, so a CI-only setup needs a target named exactly `prod`.

**Where each Databricks feature is used:**

| Feature | Use |
|---|---|
| Workspace | Stores the deployed notebook files |
| Jobs | Orchestration, task order, run history |
| Serverless compute | Runs each task with no cluster to manage |
| Unity Catalog | Schemas, tables, model registry, permissions |
| Delta tables | `telco_customers`, `telco_churn_predictions` |
| MLflow Tracking | Experiments, parameters, metrics |
| UC Model Registry | Model versions and the `Champion` alias |
| Model Serving | The real-time endpoint |

---

## 4. Prerequisites

- An Azure account
- A GitHub account and Git installed
- A laptop with PowerShell, Terminal or a similar shell
- About 2 to 3 hours the first time
- Cost: cents to a few dollars, because everything is serverless and scales to zero. The serving endpoint can still cost money while it exists, so delete it when finished.

---

## 5. Setup, step by step

### Step 1: Budget alert
1. Sign in at portal.azure.com.
2. Go to **Cost Management -> Budgets -> Add**. Create a $10 to $20 budget with email alerts at 50% and 90%. Do this first.

### Step 2: Create the Azure Databricks workspace
1. Search for **Azure Databricks -> Create**.
2. Create a resource group, for example `rg-mlops-learning`.
3. Choose a region near you that supports serverless.
4. Choose the **Premium** pricing tier.
5. Set **Workspace type** to **Serverless**. No VNet or advanced networking is needed.
6. Click **Review + create**, then **Create**.
7. After deployment open the resource, click **Launch Workspace**, and copy the workspace URL (like `https://adb-1234567890123456.7.azuredatabricks.net`). If you recreate the workspace later, the URL changes.

### Step 3: Find your catalog and test internet access
1. In the workspace open **Catalog** and note a catalog where you can create schemas. It is usually named after the workspace with a numeric suffix (for example `mlops_learning_7405610249319530`). Use the exact name shown. This is `<your_catalog>`.
2. Create a notebook on **Serverless** and run all three lines in one cell:
   ```python
   import pandas as pd
   url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
   print(pd.read_csv(url).shape)
   ```
   Expected output: `(7043, 21)`. If it fails with a network error, your workspace blocks outbound internet. Upload the CSV to a Unity Catalog Volume instead and change the `URL` line in `train.py` to the `/Volumes/...` path.

### Step 4: Install the Databricks CLI and log in
1. Install the CLI:
   - Windows: `winget install Databricks.DatabricksCLI`
   - macOS: `brew tap databricks/tap && brew install databricks`
   - Linux: use the install script from the Databricks CLI docs
2. **Close and reopen your terminal (and fully quit and reopen VS Code if you use its terminal)** so the new `PATH` is picked up.
3. Log in:
   ```
   databricks auth login --host <your-workspace-url>
   ```
   Press Enter to accept the default profile name and complete the browser sign-in.
4. Check it works:
   ```
   databricks current-user me
   ```

### Step 5: Create the project files
Create the files in the layout from section 2. Replace `<your-workspace-url>` and `<your_catalog>` in `databricks.yaml`.

#### databricks.yaml
```yaml
bundle:
  name: mlops_telco_churn

variables:
  catalog:
    description: Unity Catalog catalog to use
  schema:
    default: mlops_dev

include:
  - resources/*.yml

targets:
  dev:
    mode: development
    default: true
    workspace:
      host: https://<your-workspace-url>
    variables:
      catalog: <your_catalog>
      schema: mlops_dev
  prod:
    workspace:
      host: https://<your-workspace-url>
      root_path: /Workspace/Shared/.bundle/${bundle.name}/${bundle.target}
    variables:
      catalog: <your_catalog>
      schema: mlops_prod
```
If you only use CI/prod, you may keep only the `prod` target (the workflow uses `-t prod`). Do not rename a target by editing just part of it: the name, the schema and the default schema must all agree.

#### resources/training_job.yml
```yaml
resources:
  jobs:
    training_pipeline:
      name: telco-churn-pipeline
      tasks:
        - task_key: train
          notebook_task:
            notebook_path: ../src/train.py
            base_parameters:
              catalog: ${var.catalog}
              schema: ${var.schema}
        - task_key: validate_promote
          depends_on:
            - task_key: train
          notebook_task:
            notebook_path: ../src/validate_promote.py
            base_parameters:
              catalog: ${var.catalog}
              schema: ${var.schema}
              min_f1: '0.55'
        - task_key: batch_inference
          depends_on:
            - task_key: validate_promote
          notebook_task:
            notebook_path: ../src/batch_inference.py
            base_parameters:
              catalog: ${var.catalog}
              schema: ${var.schema}
        - task_key: deploy_endpoint
          depends_on:
            - task_key: validate_promote
          notebook_task:
            notebook_path: ../src/endpoint.py
            base_parameters:
              catalog: ${var.catalog}
              schema: ${var.schema}
      schedule:
        quartz_cron_expression: '0 0 6 * * ?'
        timezone_id: UTC
        pause_status: PAUSED
```
No compute is specified, so tasks run on serverless. The schedule is created paused so nothing runs (or costs money) until you choose.

#### src/train.py
```python
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
```

#### src/validate_promote.py
```python
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
```

#### src/batch_inference.py
```python
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
```

#### src/endpoint.py
Creates the endpoint if it does not exist, points it at the current `Champion` version if it does, waits until it is ready, and sends a test request. The endpoint name is `telco-churn-<schema>` with underscores replaced by hyphens (for example `telco-churn-mlops-prod`).

```python
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
```
The first deploy builds a container and typically takes several minutes, so this task is much slower than the others. `EndpointCoreConfigInput` needs `name` inside the config on newer `databricks-sdk` versions. The code above works with both the old and the new signature.

#### .gitignore
```
.databricks/
__pycache__/
venv/
```

### Step 6: Validate, deploy and run from your laptop
```
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run -t dev training_pipeline
```
Use `-t prod` instead if you only use the prod target (note that the endpoint task also runs).

Check the results in the workspace:
- **Jobs & Pipelines**: the run, with all tasks green.
- **Experiments**: the logged run with `accuracy`, `recall`, `f1`, `roc_auc`.
- **Catalog -> your catalog -> mlops_dev (or mlops_prod) -> Models**: `telco_churn_classifier` with a `Champion` alias.
- **Catalog -> ... -> Tables**: `telco_customers` and `telco_churn_predictions`.
- **Serving**: the endpoint, once `deploy_endpoint` finishes.

If `bundle deploy` says `Resources: 0 created`, check that your job file ends in `.yml` and sits in `resources/`.

A `validate` warning that `/Workspace/Shared/...` is writable by all workspace users is harmless for this exercise. In real projects deploy to a restricted folder or add a `permissions` block.

### Step 7: Create a service principal for CI
CI must not use your personal login.
1. Go to **Settings -> Identity and access -> Service principals -> Add service principal -> Add new** (Databricks managed). Name it `github-ci`.
2. On its **Configuration** tab, enable **Workspace access** and **Databricks SQL access**.
3. On the **Secrets** tab choose **Generate secret**. Set a lifetime such as 30 days and the scope `all-apis` (Read + Write). Copy the **secret** (shown only once) and the **Client ID**.
4. Grant it catalog access: **Catalog -> your catalog -> Permissions -> Grant**, select `github-ci`, tick **USE CATALOG** and **CREATE SCHEMA**, and confirm. Or run:
   ```sql
   GRANT USE CATALOG, CREATE SCHEMA ON CATALOG <your_catalog> TO `<service-principal-client-id>`;
   ```
5. In GitHub go to **Settings -> Secrets and variables -> Actions** and add two repository secrets:
   - `DATABRICKS_CLIENT_ID`
   - `DATABRICKS_CLIENT_SECRET`
6. If the `deploy_endpoint` task fails with a permission error, check the service principal's entitlements for serving endpoints.

Never commit these values to the repo. If you recreate the workspace, create a new service principal and replace both GitHub secrets, because the old ones belong to the deleted workspace.

### Step 8: GitHub Actions workflow
`.github/workflows/deploy.yaml`:
```yaml
name: mlops-pipeline

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

env:
  DATABRICKS_CLIENT_ID: ${{ secrets.DATABRICKS_CLIENT_ID }}
  DATABRICKS_CLIENT_SECRET: ${{ secrets.DATABRICKS_CLIENT_SECRET }}

jobs:
  validate:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: databricks/setup-cli@main
      - run: databricks bundle validate -t prod

  deploy-and-run:
    if: github.event_name == 'push'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: databricks/setup-cli@main
      - run: databricks bundle deploy -t prod
      - run: databricks bundle run -t prod training_pipeline
```

| You do | What runs |
|---|---|
| Push to a feature branch (no PR) | Nothing |
| Open or update a PR into `main` | `validate` only (no compute, no cost) |
| Merge the PR, or push directly to `main` | `deploy-and-run` (deploy to prod, then run the pipeline) |

- For real projects, pin `setup-cli` to a released version instead of `main`.
- **Azure DevOps:** put the same commands (install CLI, `bundle deploy`, `bundle run`) in `azure-pipelines.yml` and store the secrets as pipeline variables.
- Re-running a failed job re-uses the same commit. To pick up a fix, push a new commit.

### Step 9: Trigger the full automation
Create the repo on GitHub, put a first commit on `main` (so the workflow does not run before secrets exist), then push your work on a branch:
```
git init
git checkout -b main
echo "# mlops-telco-churn" > README.md
git add README.md .gitignore
git commit -m "Initial commit"
git remote add origin https://github.com/<your-username>/mlops-telco-churn.git
git push -u origin main

git checkout -b first-pipeline
git add .
git commit -m "Add telco churn MLOps pipeline"
git push -u origin first-pipeline
```
Then:
1. Open a pull request from `first-pipeline` into `main`. Watch **validate** pass.
2. Merge it. Watch **deploy-and-run** in the **Actions** tab.
3. In Databricks confirm a new run of `telco-churn-pipeline` (no `[dev ...]` prefix), and in `mlops_prod` a `Champion` model, both tables, and a serving endpoint.

### Demonstrate the quality gate
1. On a new branch, change `min_f1` to `'0.70'` in `resources/training_job.yml` (the real F1 is about 0.61, so this fails).
2. Open a PR and merge it. Watch the pipeline fail at **validate_promote**. The Champion is unchanged and the tasks after it are skipped.
3. Revert the change and merge again.

---

## 6. How to verify batch inference

Run these in a notebook on Serverless (no SQL warehouse needed). Change `mlops_prod` to `mlops_dev` to check dev.

```python
# 1. Row count: expect 7043
spark.sql("SELECT COUNT(*) AS `rows` FROM <your_catalog>.mlops_prod.telco_churn_predictions").show()

# 2. Predictions vs actual: expect four rows, no nulls
spark.sql("""
SELECT predicted_churn, actual_churn, COUNT(*) AS customers
FROM <your_catalog>.mlops_prod.telco_churn_predictions
GROUP BY predicted_churn, actual_churn
ORDER BY 1, 2
""").show()

# 3. Freshness: the newest version should match your run time
spark.sql("DESCRIBE HISTORY <your_catalog>.mlops_prod.telco_churn_predictions") \
     .select("version", "timestamp", "operation").show(truncate=False)
```

**Reading the result (example):**

| | Actually stayed (0) | Actually churned (1) |
|---|---|---|
| **Predicted stay (0)** | 3785 | 371 |
| **Predicted churn (1)** | 1389 | 1498 |

That is about 80% recall and about 52% precision. The model is tuned to catch most churners at the cost of false alarms, which is why the gate uses F1.

**Caveats:**
- The prod schema is created and owned by the service principal. To read it with your own account, grant yourself access once:
  ```sql
  GRANT USE SCHEMA, SELECT, EXECUTE ON CATALOG <your_catalog> TO `<your-email>`;
  ```
- `batch_inference` scores all customers, including the ones the model trained on, so do not read quality from this table. The honest quality number is the `f1` on the held-out test split in the MLflow experiment.

---

## 7. How to call the real-time endpoint

The endpoint is `telco-churn-mlops-prod` (name follows the schema). Its URL is:

```
https://<your-workspace-url>/serving-endpoints/telco-churn-mlops-prod/invocations
```

### Test in the UI (no token)
1. Go to **Serving -> telco-churn-mlops-prod** and confirm the state is **Ready**.
2. Click **Use -> Query**, paste this into the Request box, scroll down, and click **Send request**:
   ```json
   {"dataframe_records": [{"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85}]}
   ```
3. The response should be `{"predictions": [1]}` (1 = predicted churn, 0 = predicted to stay). The response header names the served model version.
4. Try a customer who should stay (two-year contract, long tenure, for example `"Contract": "Two year"`, `"tenure": 60`) and expect `0`.

All 19 feature columns are required, with exactly the names above.

### Call it from your laptop (PowerShell)
```powershell
databricks auth token --host https://<your-workspace-url>
```
Copy the `access_token`, then:
```powershell
$token = "<access_token>"
$body = '{"dataframe_records": [{"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85}]}'
Invoke-RestMethod -Method Post `
  -Uri "https://<your-workspace-url>/serving-endpoints/telco-churn-mlops-prod/invocations" `
  -Headers @{Authorization = "Bearer $token"} `
  -ContentType "application/json" -Body $body
```

### Call it from Python: `test_endpoint.py`
Set `DATABRICKS_TOKEN` to the access token first (`$env:DATABRICKS_TOKEN = "<access_token>"` in PowerShell).
```python
import os
import requests

HOST = 'https://<your-workspace-url>'
ENDPOINT = 'telco-churn-mlops-prod'
TOKEN = os.environ['DATABRICKS_TOKEN']

record = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}

r = requests.post(
    f'{HOST}/serving-endpoints/{ENDPOINT}/invocations',
    headers={'Authorization': f'Bearer {TOKEN}'},
    json={'dataframe_records': [record]},
    timeout=60,
)
r.raise_for_status()
print(r.json())   # {'predictions': [0 or 1]}
```

### Giving access to customers
Do not give customers the endpoint URL or a Databricks token. A token grants access to your workspace, and the endpoint requires one on every call. Put your own small API in front of it. The customer calls your API with an API key, and your API calls Databricks as a service principal. The example below has not been run, so test it before relying on it.

```python
import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from databricks.sdk import WorkspaceClient

app = FastAPI()
ENDPOINT = 'telco-churn-mlops-prod'
API_KEYS = set(os.environ['CUSTOMER_API_KEYS'].split(','))
w = WorkspaceClient()   # reads DATABRICKS_HOST, DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post('/predict')
def predict(customer: Customer, x_api_key: str = Header(...)):
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail='Invalid API key')
    resp = w.serving_endpoints.query(name=ENDPOINT, dataframe_records=[customer.model_dump()])
    return {'churn_risk': 'high' if resp.predictions[0] == 1 else 'low'}
```
Run it with `uvicorn app:app` and host it somewhere such as Azure Container Apps or Azure Functions. The service principal needs **CAN QUERY** on the endpoint. The model returns only 0 or 1. If customers need a probability, log a custom model that returns `predict_proba`.

---

## 8. Cost guardrails

- Use serverless only. Never create classic all-purpose clusters.
- Keep the Azure budget alert.
- Keep the schedule `PAUSED` unless you are actively demonstrating. Setting `pause_status: UNPAUSED` runs the job daily at 06:00 UTC and costs money on every run.
- **Delete the serving endpoint** right after demos. It scales to zero but still exists and can cost money.
- Every push to `main` runs the full pipeline, including the endpoint task.

---

## 9. Teardown

1. `databricks bundle destroy -t dev` and `databricks bundle destroy -t prod` (use only the target you deployed).
2. **Delete the serving endpoint** in **Serving**. `bundle destroy` does not remove it, because the code created it, not the bundle.
3. Delete the service principal secret in Databricks (and the service principal if you do not need it).
4. Remove the two GitHub secrets, and delete the repo or disable the workflow so future pushes do not fail.
5. In the Azure portal delete the resource group (for example `rg-mlops-learning`). This removes the workspace and everything attached to it. Check first that the group contains only this project, because deletion cannot be undone.

If you recreate the workspace, the URL and the catalog name change. Update `databricks.yaml`, create a new service principal, grant it the catalog permissions, and replace both GitHub secrets before pushing.

---

## 10. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `databricks` is not recognized | The terminal was opened before the install. Close and reopen it (fully quit VS Code), or refresh `PATH` in the current session. |
| `bundle deploy` shows `Resources: 0 created` | The job file is named `.yaml` but `include` expects `*.yml`. Rename it to `training_job.yml`. |
| `bundle run`: resource not found | The deploy did not create the job. Fix the line above and redeploy. |
| `Error: prod: no such target. Available targets: dev` | The workflow runs `-t prod` but `databricks.yaml` has no target named `prod`. Rename the target, the schema (`mlops_prod`) and the default schema to match, then push a new commit. |
| `bundle validate` errors | The message names the file and line. Update the CLI first, since bundle syntax changes between versions. |
| `NameError: name 'pd' is not defined` in the internet test | The notebook cell is missing the `import` and `url` lines. Run all three lines in one cell. |
| Download fails in `train` | The workspace blocks outbound internet. Use the Volume fallback from Step 3. |
| Permission denied creating a schema | The user or service principal lacks `CREATE SCHEMA` on the catalog. |
| `Principal ... does not have USE SCHEMA on mlops_prod` | The service principal owns the prod schema. Grant your own user `USE SCHEMA, SELECT, EXECUTE` on the catalog (see section 6). |
| `Alias not found` in `batch_inference` | The gate failed earlier, so no Champion exists yet. |
| Gate failed with an F1 below the minimum | Check the F1 in **Experiments** and set `min_f1` just under your observed value. |
| CI cannot authenticate | Check the two GitHub secrets, that the secret has not expired, and that the service principal has workspace access. |
| Tasks fail to start on serverless | Confirm serverless is enabled for your region and workspace. |
| No SQL warehouse available | Run the check queries in a Serverless notebook with `spark.sql(...)`, or create a 2X-Small serverless warehouse with auto-stop. |
| `EndpointCoreConfigInput.__init__() missing ... 'name'` | Newer `databricks-sdk` needs `name` inside the config. Use `EndpointCoreConfigInput(name=ENDPOINT, served_entities=[entity])` as in `endpoint.py`. |
| `deploy_endpoint` permission error | The service principal needs permission to create serving endpoints. Check its entitlements. |
| Endpoint stays "Not ready" | Open the endpoint's **Logs** tab for the build error. The first build takes several minutes. |
| Endpoint call returns 401/403 | The token expired (CLI tokens last about an hour) or lacks **CAN QUERY** on the endpoint. Get a new token. |
| Endpoint call returns 400 | A feature column is missing or misspelled. All 19 are required. |
| Very slow first request | Normal after scale to zero. Retry. |
| A fix does not take effect when you re-run the failed job | A re-run re-uses the same commit. Push a new commit. |

---

## 11. Extensions

- Unit tests for the gate logic, run in CI before deploy.
- Separate dev and prod workspaces, with a prod service principal as `run_as`.
- Scoring genuinely new customers, and comparing predictions to later outcomes.
- Model monitoring and a human approval step before promotion.
- A narrower service principal scope than `all-apis`.
- Defining the endpoint in the bundle itself, so `bundle destroy` removes it too.
- Databricks MLOps Stacks (`databricks bundle init mlops-stacks`) as a production-grade template.

> Note: Databricks CLI, SDK and bundle behaviour change between versions. If a command or YAML field is rejected, the CLI error message and the current Databricks documentation take precedence.
