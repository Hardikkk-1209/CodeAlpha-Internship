# CodeAlpha Data Science Internship Tasks

This repository contains three projects from the CodeAlpha Data Science task list. I chose projects that cover classification, regression, exploratory data analysis, and practical model evaluation.

## Projects

| Task | What it does | Main concepts |
|---|---|---|
| [Task 1 — Iris Flower Classification](task1_iris_classification/) | Classifies an Iris flower from four measurements | Train/test split, scaling, classification, accuracy and confusion matrix |
| [Task 2 — Car Price Prediction](task2_car_price_prediction/) | Predicts a car's selling price from common car features | Data cleaning, feature engineering, one-hot encoding, regression metrics |
| [Task 3 — Sales Prediction](task3_sales_prediction/) | Studies advertising spend and predicts product sales | Exploratory analysis, regression, feature importance, residual checks |

The projects use small, well-known or openly available datasets so they are straightforward to run and understand. They do not need API keys.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Run each project from the repository root:

```bash
python task1_iris_classification/iris_classification.py
python task2_car_price_prediction/car_price_prediction.py
python task3_sales_prediction/sales_prediction.py
```

The scripts create plots and save them inside each task's `outputs/` folder. The Iris script uses scikit-learn's built-in dataset. The car-price and sales scripts try to download the source datasets; if there is no internet connection, each script explains where to put a local CSV instead.

## Notes about data and results

- Results can vary slightly with package versions or dataset updates.
- The car price script checks for common target column names and reports an actionable error if the downloaded schema changes.
- The sales dataset is commonly known as the Advertising dataset (TV, radio, newspaper advertising spend and sales). The model estimates relationships in that dataset; it does not prove advertising caused a particular change in sales.
- Model scores should be read alongside the train/test split and the evaluation metrics, not as a guarantee of future performance.

## Folder layout

```text
codealpha_tasks/
├── README.md
├── requirements.txt
├── task1_iris_classification/
│   ├── README.md
│   └── iris_classification.py
├── task2_car_price_prediction/
│   ├── README.md
│   └── car_price_prediction.py
└── task3_sales_prediction/
    ├── README.md
    └── sales_prediction.py
```

These are learning projects. Review the code and make sure you can explain the choices, data, limitations, and results before presenting them.
