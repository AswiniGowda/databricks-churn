# Telco Customer Churn: End-to-End MLOps on Azure Databricks

A small, low-cost, fully automated MLOps pipeline. Nobody opens a notebook: a merge to `main` trains a model, checks its quality, promotes it only if it passes, and scores every customer.

```
git push -> GitHub Actions -> databricks bundle deploy -> serverless job:
            train -> validate_promote -> batch_inference
```

**Stack:** Azure Databricks (serverless), Databricks Asset Bundles, MLflow, Unity Catalog, Delta tables, GitHub Actions.

---

## 1. What this project does

| Task | What it does |
|---|---|
| `train` | Downloads the IBM Telco Customer Churn CSV, cleans it, saves it as a Delta table, trains a logistic regression pipeline, logs metrics to MLflow, and registers the model in Unity Catalog. |
| `validate_promote` | Quality gate. The new model becomes the `Champion` alias only if its F1 score is at least `min_f1` and at least the current Champion's F1. Otherwise the job fails and the Champion stays unchanged. |
| `batch_inference` | Loads the model by the `Champion` alias, scores every customer, and writes predictions to a Delta table. |

**Why F1 and not accuracy:** only about a quarter of customers churn, so a model that always predicts "no churn" would look about 73% accurate. F1 punishes that.

**Example result from a real run:** F1 0.614, recall 0.783, accuracy 0.738, ROC AUC 0.841. Your numbers may differ slightly by library version.

**Dataset:** the IBM Telco Customer Churn sample dataset (7,043 rows, 21 columns). It is a public sample dataset about a fictional telecom company.

---

## 2. Repository layout

```
mlops-telco-churn/
├─ databricks.yaml                 # bundle settings: variables, dev and prod targets
├─ resources/
│  └─ training_job.yml             # the job: 3 tasks, order, parameters, paused schedule
├─ src/
│  ├─ train.py                     # task 1
│  ├─ validate_promote.py          # task 2 (quality gate)
│  └─ batch_inference.py           # task 3
├─ .github/workflows/
│  └─ deploy.yaml                  # CI/CD: validate on PR, deploy and run on merge
├─ .gitignore
└─ README.md
```

**Important:** `databricks.yaml` includes `resources/*.yml`. The job file must end in `.yml`, not `.yaml`, or the bundle will silently find no job.

---

## 3. How it works (technical overview)

1. `databricks bundle deploy` uploads `src/*.py` into the workspace and creates a **Job** from the YAML files. The first line of each Python file, `# Databricks notebook source`, makes Databricks treat it as a notebook.
2. `databricks bundle run` calls the Jobs API to start the job. Databricks runs the three tasks in order, each on its own serverless compute. The run is labelled **Manually** because it was started by an API call, not by the schedule.
3. Parameters from `base_parameters` reach the notebooks as widgets (`dbutils.widgets.get(...)`).
4. `train` passes the model version number to the next task with `dbutils.jobs.taskValues`.
5. `validate_promote` reads the metrics from MLflow and sets the `Champion` alias, or raises an exception to fail the job.
6. `batch_inference` loads `models:/<catalog>.<schema>.telco_churn_classifier@Champion`, so it never hard-codes a version.

**dev vs prod:** both targets point at the same workspace but use different schemas (`mlops_dev`, `mlops_prod`) and different deployment paths. In a real company these would be separate workspaces.

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

---

## 4. Prerequisites

- An Azure account
- A GitHub account and Git installed
- A laptop with PowerShell, Terminal or a similar shell
- About 2 to 3 hours the first time
- Cost: cents to a few dollars, because everything is serverless and nothing idles

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
7. After deployment open the resource, click **Launch Workspace**, and copy the workspace URL (like `https://adb-1234567890123456.7.azuredatabricks.net`).

### Step 3: Find your catalog and test internet access
1. In the workspace open **Catalog** and note a catalog where you can create schemas. It is often named after the workspace with underscores (for example `mlops_learning`). This is `<your_catalog>`.
2. Create a notebook on **Serverless** and run:
   ```python
   import pandas as pd
   url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
   print(pd.read_csv(url).shape)
   ```
   Expected output: `(7043, 21)`. If it fails, your workspace blocks outbound internet. Upload the CSV to a Unity Catalog Volume instead and change the `URL` line in `train.py` to the `/Volumes/...` path.

### Step 4: Install the Databricks CLI and log in
1. Install the CLI:
   - Windows: `winget install Databricks.DatabricksCLI`
   - macOS: `brew tap databricks/tap && brew install databricks`
   - Linux: use the install script from the Databricks CLI docs
2. **Close and reopen your terminal (and VS Code if you use its terminal)** so the new `PATH` is picked up.
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

#### .gitignore
```
.databricks/
__pycache__/
venv/
```

### Step 6: Validate, deploy and run in dev from your laptop
```
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run -t dev training_pipeline
```
Check the results in the workspace:
- **Jobs & Pipelines**: the run, with all three tasks green.
- **Experiments**: the logged run with `accuracy`, `recall`, `f1`, `roc_auc`.
- **Catalog -> your catalog -> mlops_dev -> Models**: `telco_churn_classifier` with a `Champion` alias.
- **Catalog -> mlops_dev -> Tables**: `telco_customers` and `telco_churn_predictions`.

If `bundle deploy` says `Resources: 0 created`, check that your job file ends in `.yml` and sits in `resources/`.

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

Never commit these values to the repo.

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
- On a **pull request**, CI only validates the bundle (free, no compute).
- On a **merge to main**, CI deploys to prod and runs the pipeline.
- For real projects, pin `setup-cli` to a released version instead of `main`.
- You may see a warning that `/Workspace/Shared/...` is writable by all workspace users. It is harmless for this exercise. In real projects deploy to a restricted folder or add a `permissions` block.
- **Azure DevOps:** put the same commands (install CLI, `bundle deploy`, `bundle run`) in `azure-pipelines.yml` and store the secrets as pipeline variables.

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
3. In Databricks confirm a new run of `telco-churn-pipeline` (no `[dev ...]` prefix) and the `mlops_prod` schema with a `Champion` model and both tables.

This is the point of the exercise: nobody opened a notebook, yet a model was trained, gated, registered, promoted and scored.

### Demonstrate the quality gate
1. On a new branch, change `min_f1` to `'0.70'` in `resources/training_job.yml` (the real F1 is about 0.61, so this fails).
2. Open a PR and merge it. Watch the pipeline fail at **validate_promote**. The Champion is unchanged and `batch_inference` is skipped.
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

Also check that the model `telco_churn_classifier` has a `Champion` alias in **Catalog -> Models**.

**Caveats:**
- The prod schema is created and owned by the service principal. To read it with your own account, grant yourself access once:
  ```sql
  GRANT USE SCHEMA, SELECT, EXECUTE ON CATALOG <your_catalog> TO `<your-email>`;
  ```
- `batch_inference` scores all customers, including the ones the model trained on, so do not read quality from this table. The honest quality number is the `f1` on the held-out test split in the MLflow experiment.

---

## 7. Optional: real-time serving endpoint

The pipeline above is **batch** scoring and creates no endpoint or URL. A real-time endpoint is only worth it if an application needs one customer's score immediately.

**Manual test:**
1. Go to **Serving -> Create serving endpoint**.
2. Select the model `<your_catalog>.mlops_prod.telco_churn_classifier` and the version with the `Champion` alias.
3. Choose the smallest size and enable **scale to zero**.
4. In the Query panel send:
   ```json
   {"dataframe_records": [{"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85}]}
   ```
5. **Delete the endpoint** when you are done. It can still cost money while it exists.

The endpoint URL requires a Databricks access token on every call. Do not share the URL or a token with customers. Put your own API in front of it.

---

## 8. Cost guardrails

- Use serverless only. Never create classic all-purpose clusters.
- Keep the Azure budget alert.
- Keep the schedule `PAUSED` unless you are actively demonstrating. Setting `pause_status: UNPAUSED` runs the job daily at 06:00 UTC and costs money on every run.
- Delete serving endpoints right after demos.

---

## 9. Teardown

1. `databricks bundle destroy -t dev` and `databricks bundle destroy -t prod`.
2. Delete any serving endpoint (`bundle destroy` does not remove endpoints created by code).
3. Delete the service principal secret in Databricks (and the service principal if you do not need it).
4. In the Azure portal delete the resource group (for example `rg-mlops-learning`). This removes the workspace and everything attached to it.
5. Remove the two GitHub secrets.

---

## 10. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `databricks` is not recognized | The terminal was opened before the install. Close and reopen it (fully quit VS Code), or refresh `PATH` in the current session. |
| `bundle deploy` shows `Resources: 0 created` | The job file is named `.yaml` but `include` expects `*.yml`. Rename it to `training_job.yml`. |
| `bundle run`: resource not found | The deploy did not create the job. Fix the line above and redeploy. |
| `bundle validate` errors | The message names the file and line. Update the CLI first, since bundle syntax changes between versions. |
| Download fails in `train` | The workspace blocks outbound internet. Use the Volume fallback from Step 3. |
| Permission denied creating a schema | The user or service principal lacks `CREATE SCHEMA` on the catalog. |
| `Principal ... does not have USE SCHEMA on mlops_prod` | The service principal owns the prod schema. Grant your own user `USE SCHEMA, SELECT, EXECUTE` on the catalog (see section 6). |
| `Alias not found` in `batch_inference` | The gate failed earlier, so no Champion exists yet. |
| Gate failed with an F1 below the minimum | Check the F1 in **Experiments** and set `min_f1` just under your observed value. |
| CI cannot authenticate | Check the two GitHub secrets, that the secret has not expired, and that the service principal has workspace access. |
| Tasks fail to start on serverless | Confirm serverless is enabled for your region and workspace. |
| No SQL warehouse available | Run the check queries in a Serverless notebook with `spark.sql(...)`, or create a 2X-Small serverless warehouse with auto-stop. |

---

## 11. Extensions

- Unit tests for the gate logic, run in CI before deploy.
- Separate dev and prod workspaces, with a prod service principal as `run_as`.
- Scoring genuinely new customers, and comparing predictions to later outcomes.
- Model monitoring and a human approval step before promotion.
- A narrower service principal scope than `all-apis`.
- Databricks MLOps Stacks (`databricks bundle init mlops-stacks`) as a production-grade template.

> Note: Databricks CLI and bundle behaviour change between versions. If a command or YAML field is rejected, the CLI error message and the current Databricks documentation take precedence.
