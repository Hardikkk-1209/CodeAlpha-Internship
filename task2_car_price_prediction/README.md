# Task 2 — Car Price Prediction

## Goal
Train a regression model that estimates a car's price from available car-related attributes.

## What the script does
- Loads `data/car_price.csv` if you have a local copy, otherwise tries the configured public dataset URL.
- Removes duplicate rows and rows with missing or non-positive prices.
- Separates the price target from model inputs and ignores obvious index-like columns.
- Fills missing numeric and categorical values, scales numeric values, and one-hot encodes text categories using a preprocessing pipeline.
- Trains a random-forest regressor and reports MAE, RMSE and R² on held-out test data.
- Saves actual-versus-predicted and prediction-error plots plus a CSV with test predictions.

## Run
```bash
python task2_car_price_prediction/car_price_prediction.py
```

If the download fails, place the dataset CSV at `task2_car_price_prediction/data/car_price.csv`. The script looks for a target column named `selling_price`, `price`, `selling price`, or `target`.

## Understanding the metrics
- **MAE**: average absolute prediction error, expressed in the dataset's price units.
- **RMSE**: similar to MAE, but penalises large errors more strongly.
- **R²**: compares predictions with a baseline that predicts the test-set mean; it can be negative.

Prices depend on location, time, condition, and other factors that may not be included in the dataset. This is a learning model, not a valuation guarantee.
