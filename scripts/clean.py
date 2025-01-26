import pandas as pd
import numpy as np
import sklearn

# Open file with path
file_path = "C:\\Users\\jsten\\PycharmProjects\\PythonProject_test\\data\\cmu-sleep.csv"
# Import the data set
df = pd.read_csv(file_path)

# Drop unecessary columns
df = df.drop(columns=['subject_id', 'study', 'cohort', 'term_units'])

# Change types
for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# List of binary columns to process
binary_columns = ['demo_race', 'demo_gender', 'demo_firstgen']

# Replace non-binary values (not 0 or 1) with NaN
for col in binary_columns:
    df[col] = df[col].where(df[col].isin([0, 1]), np.nan)

# Replace NaN in the Zterm_units_ZofZ column for zero

df['Zterm_units_ZofZ'] = df['Zterm_units_ZofZ'].fillna(0)

## Drop rows with any NaN values
df= df.dropna()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Define features (X) and target (y)
X = df.drop(columns=['term_gpa'])
y = df['term_gpa']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Create a Pipeline with Preprocessing + Model
pipeline = Pipeline([
    ('rf_model', RandomForestRegressor(random_state=42))  # RandomForest Model
])

# Step 2: Hyperparameter Grid for Random Forest Tuning
param_grid = {
    'rf_model__n_estimators': [50, 100, 150],  # Number of trees
    'rf_model__max_depth': [None, 10, 20],    # Maximum depth of trees
    'rf_model__min_samples_split': [2, 5, 10],  # Minimum samples to split an internal node
    'rf_model__min_samples_leaf': [1, 2, 4],    # Minimum samples at each leaf node
    'rf_model__max_features': ['sqrt', 'log2'], # Number of features to consider when looking for the best split
}

# Step 3: Repeated K-Fold Cross-Validation Setup
cv = RepeatedKFold(n_splits=40, n_repeats=10, random_state=42)

# Step 4: Perform Grid Search with Repeated K-Fold Cross-Validation
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Step 5: Extract the Best Model
best_model = grid_search.best_estimator_

# Step 6: Calculate MSE for Training and Test Sets
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Step 7: Print Results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Mean Cross-Validation MSE (CV): {-grid_search.best_score_:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# Step 8: Extract and Print Feature Importances
feature_importances = best_model.named_steps['rf_model'].feature_importances_

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

# Optional: Visualizing Cross-Validation Results
cv_results = -grid_search.cv_results_['mean_test_score']  # Convert to positive MSE
plt.plot(cv_results, label='Cross-Validation MSE')
plt.title('Model Performance Across Hyperparameter Combinations')
plt.ylabel('MSE')
plt.xlabel('Hyperparameter Combination Index')
plt.legend()
plt.show()

import joblib

joblib.dump(best_model, "best_random_forest_model.pkl")

loaded_model = joblib.load("best_random_forest_model.pkl")