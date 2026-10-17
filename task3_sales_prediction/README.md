# Task 3 — Sales Prediction using Python

## Goal
Study the relationship between sales and advertising spend across TV, radio and newspaper channels, then train a regression model to estimate sales.

## What the script does
1. Loads the Advertising dataset from a public CSV URL or from a local CSV.
2. Cleans column names, removes an unused index column and converts the four expected fields to numeric values.
3. Displays a preview and descriptive statistics.
4. Saves a correlation heatmap and advertising-spend plots.
5. Splits the data into training and test sets, trains linear regression, and reports MAE, RMSE and R².
6. Prints the fitted model coefficients and saves predictions and a comparison chart.

## Run
```bash
python task3_sales_prediction/sales_prediction.py
```

If downloading is not possible, place the CSV in `task3_sales_prediction/data/advertising.csv`. It must contain `TV`, `Radio`, `Newspaper`, and `Sales` columns (case-insensitive).

## Interpretation
The coefficients show how the fitted model associates each advertising channel with sales while accounting for the other two channels. The dataset is small and observational, so the results should not be treated as proof of causality or as a guarantee for a real marketing budget.
