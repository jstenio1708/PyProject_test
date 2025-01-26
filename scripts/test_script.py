import pandas as pd
import numpy as np
import sklearn

# File path with double backslashes
file_path = "C:\\Users\\jsten\\PycharmProjects\\PythonProject_test\\cmu-sleep.csv"
# Import the data set
df = pd.read_csv(file_path)


# Get the column names
column_names = df.columns.tolist()
print(f"Column names: {column_names}")

df = df.drop(columns=['subject_id', 'study', 'cohort', 'term_units'])

# Display the first few rows of the dataframe
print(df.head())

# Get types
print(df.dtypes)

for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Check the data types after transformation
print(df.dtypes)

# List of binary columns to process
binary_columns = ['demo_race', 'demo_gender', 'demo_firstgen']

# Replace non-binary values (not 0 or 1) with NaN
for col in binary_columns:
    df[col] = df[col].where(df[col].isin([0, 1]), np.nan)

# Check the result
print(df[binary_columns].head())

# Check na values
print(df.isna().sum())


# Create a clean Dataset for stats
df_clean = df.dropna(subset=binary_columns)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
import numpy as np


# Random Forest Imputation Function
def random_forest_imputation(df):
    df_copy = df.copy()

    for column in df_copy.columns:
        if df_copy[column].isnull().any():
            # Split into rows with and without missing data for the column
            missing_data = df_copy[df_copy[column].isnull()]
            non_missing_data = df_copy.dropna(subset=[column])

            # Define features (X) and target (y)
            X_train = non_missing_data.drop(columns=[column])
            y_train = non_missing_data[column]

            X_missing = missing_data.drop(columns=[column])

            if y_train.nunique() == 2:  # Binary classification
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # Continuous regression
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            # Train the model
            model.fit(X_train, y_train)

            # Predict missing values
            predicted_values = model.predict(X_missing)

            # Impute missing values
            df_copy.loc[df_copy[column].isnull(), column] = predicted_values

    return df_copy


# Apply the imputation function
df_imputed = random_forest_imputation(df)

# Check the result
print(df_imputed.isna().sum())  # All columns should show 0 missing values

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Ensure df_imputed is ready
# Assume df_imputed is already imputed and has no missing values

# Define features (X) and target (y)
X = df_imputed.drop(columns=['term_gpa'])
y = df_imputed['term_gpa']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model performance (optional)
train_score = rf_model.score(X_train, y_train)
test_score = rf_model.score(X_test, y_test)

print(f"Training R^2: {train_score:.2f}")
print(f"Testing R^2: {test_score:.2f}")

# Extract feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame to display feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances from Random Forest')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important at the top
plt.show()

