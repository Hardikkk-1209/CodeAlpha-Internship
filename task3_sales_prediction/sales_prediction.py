"""Explore advertising spend and estimate sales with linear regression."""

from pathlib import Path
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "justmarkham/DAT8/master/data/Advertising.csv"
)
LOCAL_CSV = DATA_DIR / "advertising.csv"


def load_data():
    if LOCAL_CSV.exists():
        print(f"Using local dataset: {LOCAL_CSV}")
        return pd.read_csv(LOCAL_CSV)

    try:
        print("Downloading the advertising dataset...")
        with urllib.request.urlopen(DATA_URL, timeout=20) as response:
            contents = response.read()
        LOCAL_CSV.write_bytes(contents)
        return pd.read_csv(LOCAL_CSV)
    except Exception as error:
        raise RuntimeError(
            "Could not download the dataset. Save the Advertising CSV as "
            f"{LOCAL_CSV} and run again. Download error: {error}"
        ) from error


def main():
    data = load_data()
    data.columns = [str(column).strip().lower() for column in data.columns]
    data = data.loc[:, ~data.columns.str.startswith("unnamed")]
    required_columns = {"tv", "radio", "newspaper", "sales"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns {sorted(missing_columns)}. "
            f"Columns found: {list(data.columns)}"
        )

    for column in required_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=list(required_columns))
    data = data[(data[["tv", "radio", "newspaper"]] >= 0).all(axis=1)]

    print("\nDataset preview:")
    print(data.head().to_string(index=False))
    print(f"\nRows after cleaning: {len(data)}")
    print("\nSummary statistics:")
    print(data[["tv", "radio", "newspaper", "sales"]].describe().round(2))

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        data[["tv", "radio", "newspaper", "sales"]].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Advertising and sales correlation")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    spend_columns = ["tv", "radio", "newspaper"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, column in zip(axes, spend_columns):
        sns.regplot(
            data=data,
            x=column,
            y="sales",
            scatter_kws={"alpha": 0.65},
            line_kws={"color": "black"},
            ax=ax,
        )
        ax.set_title(f"{column.title()} spend vs sales")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "spend_vs_sales.png", dpi=160)
    plt.close(fig)

    X = data[spend_columns]
    y = data["sales"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\nTest-set metrics:")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")
    print("\nModel coefficients:")
    for feature, coefficient in zip(spend_columns, model.coef_):
        print(f"{feature.title():10s}: {coefficient:.3f}")
    print(f"Intercept  : {model.intercept_:.3f}")

    results = pd.DataFrame(
        {"actual_sales": y_test.to_numpy(), "predicted_sales": predictions}
    )
    results.to_csv(OUTPUT_DIR / "sales_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=results, x="actual_sales", y="predicted_sales", ax=ax)
    low = min(results.min())
    high = max(results.max())
    ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
    ax.set_title("Actual vs predicted sales")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    print(f"\nPlots and prediction CSV saved to: {OUTPUT_DIR}")
    print(
        "\nInterpretation note: these results describe this dataset. "
        "They do not prove that increasing an advertising channel will "
        "always cause sales to rise."
    )


if __name__ == "__main__":
    main()
