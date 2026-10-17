"""Classify Iris flowers using a simple, reproducible ML workflow."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    # Scikit-learn provides this classic dataset, so there is no download step.
    iris = load_iris(as_frame=True)
    features = iris.data
    labels = iris.target

    data = features.copy()
    data["species"] = labels.map(dict(enumerate(iris.target_names)))
    print("\nFirst five rows:")
    print(data.head().to_string(index=False))
    print("\nRows per species:")
    print(data["species"].value_counts().to_string())

    # A stratified split keeps all three species represented in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    # Scaling is fitted only on training data by using a pipeline.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nTest accuracy: {accuracy:.3f} ({accuracy:.1%})")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=iris.target_names,
            zero_division=0,
        )
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=iris.target_names,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Iris classification — test set")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    plot_data = data.copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=plot_data,
        x="petal length (cm)",
        y="petal width (cm)",
        hue="species",
        palette="deep",
        s=60,
        ax=ax,
    )
    ax.set_title("Petal measurements by Iris species")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "petal_measurements.png", dpi=160)
    plt.close(fig)

    print(f"\nPlots saved to: {OUTPUT_DIR}")
    print("Example input: sepal length, sepal width, petal length, petal width (cm).")
    example = [[5.1, 3.5, 1.4, 0.2]]
    predicted_id = model.predict(example)[0]
    print(f"Example prediction: {iris.target_names[predicted_id]}")


if __name__ == "__main__":
    main()
