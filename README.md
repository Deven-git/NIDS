# Explainable Ensemble Machine Learning-Based Network Intrusion Detection System

Using Feature Selection and Class Balancing

## Overview

This project is a web-based Network Intrusion Detection System (NIDS) that
classifies network traffic into normal (Benign) and 8 attack categories using
an ensemble of machine learning models. It addresses common limitations in
existing IDS research by combining rigorous multi-model comparison, class
balancing for rare attacks, feature selection for efficiency, and SHAP-based
explainability — all delivered through a working Flask web application.

## Key Features

- **7-model comparison**: Decision Tree, Random Forest, XGBoost, LightGBM,
  CatBoost, SVM, and KNN, evaluated on Accuracy, Precision, Recall, F1-Score,
  and ROC-AUC (not accuracy alone).
- **Class balancing**: SMOTE + undersampling hybrid strategy to address
  severe class imbalance (Benign traffic outnumbers Heartbleed 186,000:1
  in the raw dataset).
- **Feature selection**: Correlation filtering + Random Forest importance
  ranking, reducing 69 raw features to 30 without meaningful accuracy loss.
- **Explainable AI**: SHAP (SHapley Additive exPlanations) provides both
  global (which features matter overall) and local (why this specific
  prediction) explanations.
- **Full web application**: Login, CSV upload with validation, live
  prediction, interactive dashboard, model comparison table, SHAP browser,
  and downloadable PDF reports.

## Dataset

[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) — 2.83 million
network flow records across 8 CSV files, covering Benign traffic and 8
attack categories: DDoS, DoS, Port Scan, Botnet, Brute Force, Web Attack,
Infiltration, and Heartbleed.

## Results Summary

| Model | Accuracy | F1-Score (macro) | ROC-AUC |
|---|---|---|---|
| **CatBoost (production model)** | **99.87%** | **96.22%** | **99.97%** |
| Random Forest | 99.87% | 94.37% | 99.03% |
| LightGBM | 99.89% | 93.82% | 100.0% |
| XGBoost | 99.88% | 89.80% | 100.0% |
| Decision Tree | 99.82% | 82.92% | 98.08% |
| KNN | 97.13% | 61.32% | 97.66% |
| SVM | 92.35% | 57.10% | 99.38% |

CatBoost was selected as the production model for its combination of
top-tier F1-score (fair performance across both common and rare attack
classes) and fastest prediction time among top performers — critical for
real-time deployment.

## Project Structure

```
nids-project/
├── data/                   # raw, interim, and processed datasets
├── src/
│   ├── preprocessing/       # cleaning, splitting, balancing scripts
│   ├── feature_selection/
│   ├── models/               # training, tuning scripts
│   ├── evaluation/
│   └── explainability/        # SHAP generation
├── saved_models/              # trained model artifacts (.pkl)
├── reports/                    # generated figures, comparison tables
├── app/                          # Flask web application
│   ├── routes/                    # blueprints (auth, upload, predict, etc.)
│   ├── templates/
│   ├── static/
│   └── database/
├── docs/                            # this documentation
├── requirements.txt
└── run.py                            # Flask entry point
```

## Quick Start

See [INSTALLATION.md](INSTALLATION.md) for full setup instructions. Summary:

```bash
# 1. Set up environment
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Run the ML pipeline (in order)
python src/preprocessing/load_data.py
python src/preprocessing/clean_data.py
python src/feature_selection/select_features.py
python src/preprocessing/split_and_balance.py
python src/models/train_models.py
python src/models/tune_models.py
python src/evaluation/generate_evaluation_plots.py
python src/explainability/generate_shap_explanations.py

# 3. Set up and run the web app
pip install flask-sqlalchemy flask-login reportlab
python save_feature_list.py
python copy_figures_to_static.py
python init_db.py
python run.py
```

Then visit `http://127.0.0.1:5000` and log in with `admin` / `admin123`
(change this password before any real deployment).

## Technology Stack

- **ML**: scikit-learn, XGBoost, LightGBM, CatBoost, imbalanced-learn, SHAP
- **Data**: pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Backend**: Flask, Flask-SQLAlchemy, Flask-Login
- **Frontend**: Bootstrap 5, Chart.js
- **Database**: SQLite
- **Reporting**: ReportLab (PDF generation)

## Documentation

- [Installation Guide](INSTALLATION.md)
- [Architecture & Flowchart](docs/architecture/)
- [UML & ER Diagrams](docs/uml/)
- [API Documentation](docs/api/)
- [Research Methodology](docs/research/methodology.md)
- [Results Chapter](docs/research/results.md)
- [Conclusion & Future Scope](docs/research/conclusion.md)

## License

Academic / research project — MCA final-year submission.