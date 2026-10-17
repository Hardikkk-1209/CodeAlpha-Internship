"""Estimate used-car prices with a clear preprocessing and regression pipeline."""

from pathlib import Path
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "YBI-Foundation/Dataset/main/Car%20Price.csv"
)
LOCAL_CSV = DATA_DIR / "car_price.csv"


def load_data():
    if LOCAL_CSV.exists():
        print(f"Using local dataset: {LOCAL_CSV}")
        return pd.read_csv(LOCAL_CSV)

    try:
        print("Downloading the car price dataset...")
        with urllib.request.urlopen(DATA_URL, timeout=20) as response:
            contents = response.read()
        LOCAL_CSV.write_bytes(contents)
        return pd.read_csv(LOCAL_CSV)
    except Exception as error:
        raise RuntimeError(
            "Could not download the dataset. Save your car-price CSV as "
            f"{LOCAL_CSV} and run the script again. Download error: {error}"
        ) from error


def find_target_column(columns):
    by_lower = {str(column).strip().lower(): column for column in columns}
    for candidate in ("selling_price", "price", "selling price", "target"):
        if candidate in by_lower:
            return by_lower[candidate]
    raise ValueError(
        "Could not identify the price column. Expected one of: "
        "selling_price, price, selling price, target. "
        f"Columns found: {list(columns)}"
    )


def main():
    data = load_data()
    data.columns = [str(column).strip() for column in data.columns]
    data = data.drop_duplicates().copy()

    target_column = find_target_column(data.columns)
    data[target_column] = pd.to_numeric(data[target_column], errors="coerce")
    data = data.dropna(subset=[target_column])
    data = data[data[target_column] > 0].copy()

    if len(data) < 20:
        raise ValueError("The dataset has too few usable rows to train and test a model.")

    print("\nDataset preview:")
    print(data.head().to_string(index=False))
    print(f"\nUsable rows: {len(data)}")
    print(f"Price column: {target_column}")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    id_words = ("unnamed", "index", "serial", "record_id")
    id_columns = [
        column for column in X.columns
        if any(word in column.lower() for word in id_words)
    ]
    X = X.drop(columns=id_columns, errors="ignore")

    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable feature columns were found in the dataset.")

    numeric_pipeline = Pipeline(
        steps=[
            ("fill_missing", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=250,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print("\nTest-set metrics:")
    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²  : {r2:.3f}")

    comparison = pd.DataFrame(
        {"Actual price": y_test.to_numpy(), "Predicted price": predictions}
    )
    comparison.to_csv(OUTPUT_DIR / "actual_vs_predicted.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=comparison, x="Actual price", y="Predicted price", ax=ax)
    low = min(comparison.min())
    high = max(comparison.max())
    ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
    ax.set_title("Actual vs predicted car prices")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    errors = comparison["Actual price"] - comparison["Predicted price"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(errors, bins=25, edgecolor="white")
    ax.set_title("Prediction error distribution")
    ax.set_xlabel("Actual price − predicted price")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "prediction_errors.png", dpi=160)
    plt.close(fig)

    print(f"\nComparison CSV and plots saved to: {OUTPUT_DIR}")
    print("Note: prices and metrics use the same units as the source dataset.")


if __name__ == "__main__":
    main()
