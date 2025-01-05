# US-house-price-prediction
# House Price Prediction

This repository contains a Python implementation of a house price prediction model based on the **Kaggle House Prices: Advanced Regression Techniques** dataset. The model uses features such as living area, number of bedrooms, and other property characteristics to predict house prices.

## Dataset

The project uses the datasets provided in the Kaggle competition. The following files are required:
- `train.csv`: Training data containing features and target variable (`SalePrice`).
- `test.csv`: Test data to make predictions for submission.

You can download the dataset from the [Kaggle competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

## Features Used

- `GrLivArea`: Above ground living area (in square feet).
- `BedroomAbvGr`: Number of bedrooms above ground.
- `FullBath`: Number of full bathrooms.
- `YearBuilt`: Year the house was built.

## Model

The model is implemented using **Ridge Regression**, a linear model that includes regularization to reduce overfitting.

### Steps

1. **Data Preprocessing**:
   - Missing values are handled.
   - Features are scaled using `StandardScaler`.
   - The target variable (`SalePrice`) is log-transformed to address skewness.

2. **Training**:
   - The dataset is split into training and validation sets.
   - Ridge Regression is trained on the processed training data.

3. **Evaluation**:
   - The model is evaluated using Mean Squared Error (MSE) and R² score on the validation set.

4. **Prediction**:
   - Predictions are made on the test dataset, and the results are saved as a `submission.csv` file.

## Requirements

The project requires the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`

