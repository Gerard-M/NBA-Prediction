# Import libraries
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestRegressor(random_state=42)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [None, 10, 20],     # Max depth of trees
    'min_samples_split': [2, 5],     # Min samples to split a node
    'min_samples_leaf': [1, 2],      # Min samples at each leaf
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,               # 5-fold cross-validation
    scoring='r2',       # Optimize for R² score
    n_jobs=-1,          # Use all CPU cores
    verbose=1           # Print progress
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate on test data
predictions = best_model.predict(X_test)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

# Print results
print("\n=== Best Hyperparameters ===")
print(grid_search.best_params_)

print("\n=== Model Performance ===")
print(f"R² Score: {r2:.3f}")
print(f"Mean Absolute Error (MAE): ${mae * 100000:.2f} (in USD)")