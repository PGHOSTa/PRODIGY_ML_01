import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load datasets
train_data = pd.read_csv("C:/Users/DELL/Desktop/prodigy/train.csv")
test_data = pd.read_csv("C:/Users/DELL/Desktop/prodigy/house-prices-advanced-regression-techniques/test.csv")

# Add dataset labels and combine datasets
train_data['dataset'] = 'train'
test_data['dataset'] = 'test'
test_data['SalePrice'] = np.nan
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# Handle missing values
for col in combined_data.columns:
    if combined_data[col].dtype == 'object':
        combined_data[col].fillna(combined_data[col].mode()[0], inplace=True)
    else:
        combined_data[col].fillna(combined_data[col].median(), inplace=True)

# Preserve dataset labels and encode categorical variables
dataset_labels = combined_data['dataset']
combined_data = pd.get_dummies(combined_data.drop(columns=['dataset']), drop_first=True)
combined_data['dataset'] = dataset_labels

# Separate train and test datasets
train_data = combined_data[combined_data['dataset'] == 'train'].drop(columns=['dataset'])
test_data = combined_data[combined_data['dataset'] == 'test'].drop(columns=['dataset', 'SalePrice'])

# Log-transform target
train_data['SalePrice'] = np.log1p(train_data['SalePrice'])

# Define features and target
X = train_data.drop(columns=['SalePrice', 'Id'])
y = train_data['SalePrice']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
ridge = Ridge()
param_grid = {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best model
best_ridge = grid_search.best_estimator_

# Evaluate model
y_val_pred = best_ridge.predict(X_val)
mse_val = mean_squared_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

# Visualize predictions vs actuals
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_val_pred, alpha=0.6, edgecolors='k')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linestyle='--')
plt.xlabel('Actual Log SalePrice')
plt.ylabel('Predicted Log SalePrice')
plt.title('Validation: Predicted vs Actual Log SalePrice')
plt.show()

# Print results
print(f"Validation MSE: {mse_val}")
print(f"Validation R²: {r2_val}")
print(f"Best Alpha: {grid_search.best_params_['alpha']}")

# Prepare test predictions
test_predictions = best_ridge.predict(scaler.transform(test_data.drop(columns=['Id'])))
test_predictions = np.expm1(test_predictions)  # Reverse log transformation

# Save submission file
submission = pd.DataFrame({
    "Id": test_data['Id'],
    "SalePrice": test_predictions
})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
