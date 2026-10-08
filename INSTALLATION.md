# Installation Guide

## Prerequisites

- **Python 3.11** (recommended — this project was developed and tested on
  3.11.9). Newer versions (3.13+) may lack pre-built wheels for some
  packages like CatBoost/LightGBM at time of writing, forcing slow/failing
  source builds.
- Windows, macOS, or Linux
- ~5 GB free disk space (CICIDS2017 dataset + generated artifacts)
- The [CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
  (8 CSV files)

## 1. Project Setup

```bash
mkdir nids-project && cd nids-project

mkdir -p data/raw data/interim data/processed
mkdir -p notebooks
mkdir -p src/preprocessing src/feature_selection src/models src/evaluation src/explainability src/utils
mkdir -p saved_models
mkdir -p reports/figures reports/pdf_reports
mkdir -p app/static/css app/static/js app/templates app/routes app/database
mkdir -p docs/architecture docs/uml docs/research
mkdir -p tests
```

Place the 8 CICIDS2017 CSV files into `data/raw/`.

## 2. Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

`requirements.txt`:
```
pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
imbalanced-learn
shap
matplotlib
seaborn
plotly
flask
flask-login
flask-sqlalchemy
werkzeug
reportlab
joblib
python-dotenv
```

> **Note:** Version numbers are intentionally left unpinned. Pinning exact
> versions (e.g. `matplotlib==3.9.1`) can force pip to build from source if
> no matching wheel exists for your Python version — a common cause of
> Windows build failures requiring a C compiler that isn't installed by
> default. Unpinned installs let pip select a version with an available
> pre-built wheel.

## 4. Run the ML Pipeline (in order)

Each step depends on the previous one's output. Run from the project root.

| Step | Command | Approx. time |
|---|---|---|
| 1. Load & inspect data | `python src/preprocessing/load_data.py` | 2-3 min |
| 2. Clean data | `python src/preprocessing/clean_data.py` | 1-2 min |
| 3. Feature selection | `python src/feature_selection/select_features.py` | 3-5 min |
| 4. Split + SMOTE balance | `python src/preprocessing/split_and_balance.py` | 1 min |
| 5. Train 7 models | `python src/models/train_models.py` | 15-20 min |
| 6. Hyperparameter tuning | `python src/models/tune_models.py` | 45-90 min |
| 7. Restore best configs | `python src/models/restore_best_models.py` | 3-5 min |
| 8. Generate evaluation plots | `python src/evaluation/generate_evaluation_plots.py` | 1 min |
| 9. Generate SHAP explanations | `python src/explainability/generate_shap_explanations.py` | 3-5 min |

## 5. Set Up the Web Application

```bash
pip install flask-sqlalchemy flask-login

python save_feature_list.py
python copy_figures_to_static.py
python init_db.py
python run.py
```

Visit `http://127.0.0.1:5000`. Default login: `admin` / `admin123`.

## Troubleshooting

**`meson.build ERROR: Unknown compiler(s)` during pip install**
Caused by a pinned package version with no pre-built wheel for your Python
version, forcing a source build. Fix: use unpinned `requirements.txt`
(see above), or install Python 3.11 specifically for this project.

**`ModuleNotFoundError` despite package appearing installed**
Usually means a new terminal/IDE window is using a different Python
interpreter than the one with your venv activated. Confirm with:
```bash
python -c "import sys; print(sys.executable)"
```
This should point inside your `venv` folder. If not, activate the venv in
this terminal, or point your IDE's interpreter selector at
`<venv_path>\Scripts\python.exe`.

**`ArrayMemoryError` during SMOTE**
Occurs if SMOTE's default `sampling_strategy="auto"` tries to oversample
every minority class up to match the majority class count exactly — for
severely imbalanced data (e.g. 9 Heartbleed vs. 1.6M+ Benign samples) this
can require tens of millions of rows in memory. This project uses a capped
hybrid strategy (oversample minorities to a ceiling, undersample the
majority) instead — see `split_and_balance.py`.

**LightGBM significantly underperforms other tree models**
Usually a `num_leaves`/`max_depth` mismatch — LightGBM is controlled
primarily by `num_leaves` (leaf-wise growth), unlike XGBoost/CatBoost's
depth-wise growth. Setting `max_depth` without raising `num_leaves`
proportionally starves the model of capacity. See the corrected
configuration in `train_models.py`.

**SHAP crashes with exit code `-1073741819` (access violation) on Windows**
A known fragility between certain `shap` package versions and CatBoost on
Windows. This project's `generate_shap_explanations.py` uses CatBoost's
own native `get_feature_importance(type="ShapValues")` method instead of
the external `shap` package's `TreeExplainer`, avoiding the crash entirely
while producing mathematically equivalent SHAP values.

**`RandomizedSearchCV` with CatBoost appears to hang on Windows**
A known deadlock risk when sklearn's multiprocessing (`n_jobs=-1`) combines
with CatBoost's own internal multithreading. Fix: run the search for
CatBoost specifically with `n_jobs=1` and let CatBoost use
`thread_count=-1` internally instead.