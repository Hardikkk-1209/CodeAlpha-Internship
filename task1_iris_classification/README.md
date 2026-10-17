# Task 1 — Iris Flower Classification

## Goal
Predict whether an Iris flower is setosa, versicolor, or virginica using sepal and petal measurements.

## How it works
1. Load the Iris dataset included with scikit-learn.
2. Split the observations into training and test sets while keeping species balanced.
3. Scale measurements and train logistic regression in a pipeline.
4. Report test accuracy, precision, recall and F1-score.
5. Save a confusion matrix and a petal-measurement scatter plot to `outputs/`.

## Run
From the repository root, install `requirements.txt`, then run:

```bash
python task1_iris_classification/iris_classification.py
```

## What to discuss
The confusion matrix shows which species were classified correctly and which were confused. The example prediction at the end uses measurements in centimetres. Because this is a small, familiar dataset, the score is a demonstration rather than evidence that the model will generalise to every real-world flower.
