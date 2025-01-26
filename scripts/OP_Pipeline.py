from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import sys


class PipelineStage(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def process(self, data):
        pass


class DataPreprocessing(PipelineStage):
    def __init__(self, file_path, target_column):
        super().__init__('Data Preprocessing')
        self.file_path = file_path
        self.target_column = target_column

    def process(self, data):
        print(f"[{self.name}] Loading and preprocessing data")
        try:
            # Load dataset
            df = pd.read_csv(self.file_path)
            print("Dataset loaded successfully.")
        except FileNotFoundError:
            print(f"Error: The file '{self.file_path}' was not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading the dataset: {e}")
            sys.exit(1)

        try:
            # Preserve target column and drop others
            df = df.drop(columns=['subject_id', 'study', 'cohort', 'term_units'])

            # Type conversions
            for col in df.select_dtypes(include=['object']).columns:
                if col != self.target_column:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Binary columns handling
            binary_cols = ['demo_race', 'demo_gender', 'demo_firstgen']
            for col in binary_cols:
                df[col] = df[col].where(df[col].isin([0, 1]), np.nan)

            # Handle missing values
            df['Zterm_units_ZofZ'] = df['Zterm_units_ZofZ'].fillna(0)
            df = df.dropna()

            print("Data preprocessing completed successfully.")
            return {'df': df, 'target': self.target_column}
        except KeyError as e:
            print(f"Error: A required column is missing in the dataset: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error during data preprocessing: {e}")
            sys.exit(1)


class DataSplitter(PipelineStage):
    def __init__(self, test_size=0.2, random_state=42):
        super().__init__('Data Splitting')
        self.test_size = test_size
        self.random_state = random_state

    def process(self, data):
        print(f"[{self.name}] Splitting data")
        try:
            df = data['df']
            target = data['target']

            X = df.drop(columns=[target])
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state
            )

            data.update({
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test
            })
            print("Data splitting completed successfully.")
            return data
        except Exception as e:
            print(f"Error during data splitting: {e}")
            sys.exit(1)


class ModelTrainer(PipelineStage):
    def __init__(self, param_grid, cv_splits=40, cv_repeats=10, random_state=42):
        super().__init__('Model Training')
        self.param_grid = param_grid
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        self.random_state = random_state

    def process(self, data):
        print(f"[{self.name}] Training model")
        try:
            pipeline = Pipeline([
                ('rf_model', RandomForestRegressor(random_state=self.random_state))
            ])

            cv = RepeatedKFold(
                n_splits=self.cv_splits,
                n_repeats=self.cv_repeats,
                random_state=self.random_state
            )

            grid_search = GridSearchCV(
                pipeline, self.param_grid,
                cv=cv, n_jobs=-1,
                verbose=2, scoring='neg_mean_squared_error'
            )
            grid_search.fit(data['X_train'], data['y_train'])

            data.update({
                'best_model': grid_search.best_estimator_,
                'grid_search': grid_search
            })
            print("Model training completed successfully.")
            return data
        except Exception as e:
            print(f"Error during model training: {e}")
            sys.exit(1)


class ModelEvaluator(PipelineStage):
    def __init__(self):
        super().__init__('Model Evaluation')

    def process(self, data):
        print(f"[{self.name}] Evaluating model")
        try:
            model = data['best_model']
            X_train, X_test = data['X_train'], data['X_test']
            y_train, y_test = data['y_train'], data['y_test']

            # Calculate predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            # Calculate metrics
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)

            # Calculate MAE
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)

            # Calculate MAE as a percentage of the GPA scale (0-4)
            train_mae_percentage = (train_mae / 4.0) * 100
            test_mae_percentage = (test_mae / 4.0) * 100

            # Feature importances
            importances = model.named_steps['rf_model'].feature_importances_
            importance_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            data.update({
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_mae_percentage': train_mae_percentage,
                'test_mae_percentage': test_mae_percentage,
                'importance_df': importance_df
            })
            print("Model evaluation completed successfully.")
            return data
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            sys.exit(1)


class ModelSaver(PipelineStage):
    def __init__(self, model_path):
        super().__init__('Model Saving')
        self.model_path = model_path

    def process(self, data):
        print(f"[{self.name}] Saving model")
        try:
            joblib.dump(data['best_model'], self.model_path)
            print(f"Model saved successfully to {self.model_path}.")
            return data
        except Exception as e:
            print(f"Error saving the model: {e}")
            sys.exit(1)


class MLPipeline:
    def __init__(self, stages):
        self.stages = stages

    def run(self):
        data = {}
        for stage in self.stages:
            data = stage.process(data)
        return data


if __name__ == "__main__":
    # Configuration
    config = {
        'file_path': r"C:\Users\jsten\PycharmProjects\PythonProject_test\data\cmu-sleep.csv",  # Windows path to dataset
        'target_column': 'term_gpa',
        'model_save_path': r"C:\Users\jsten\PycharmProjects\PythonProject_test\models\best_random_forest_model.pkl",  # Windows path to save model
        'cv_splits': 10,
        'cv_repeats': 5,
        'random_state': 42,
        'param_grid': {
            'rf_model__n_estimators': [50, 100, 150],
            'rf_model__max_depth': [None, 10, 20],
            'rf_model__min_samples_split': [2, 5, 10],
            'rf_model__min_samples_leaf': [1, 2, 4],
            'rf_model__max_features': ['sqrt', 'log2']
        }
    }

    # Create pipeline stages
    stages = [
        DataPreprocessing(
            file_path=config['file_path'],
            target_column=config['target_column']
        ),
        DataSplitter(random_state=config['random_state']),
        ModelTrainer(
            param_grid=config['param_grid'],
            cv_splits=config['cv_splits'],
            cv_repeats=config['cv_repeats'],
            random_state=config['random_state']
        ),
        ModelEvaluator(),
        ModelSaver(config['model_save_path'])
    ]

    # Run pipeline
    pipeline = MLPipeline(stages)
    results = pipeline.run()

    # Save cleaned dataset to CSV
    try:
        cleaned_data_path = r"C:\Users\jsten\PycharmProjects\PythonProject_test\models\cleaned_data.csv"
        results['df'].to_csv(cleaned_data_path, index=False)
        print(f"Cleaned dataset saved to {cleaned_data_path}.")
    except Exception as e:
        print(f"Error saving cleaned dataset to CSV: {e}")

    # Save model performance statistics to CSV
    try:
        stats_path = r"C:\Users\jsten\PycharmProjects\PythonProject_test\models\model_performance_stats.csv"
        stats_df = pd.DataFrame({
            'Metric': ['Train MSE', 'Test MSE', 'Train MAE', 'Test MAE', 'Train MAE %', 'Test MAE %'],
            'Value': [
                results['train_mse'], results['test_mse'],
                results['train_mae'], results['test_mae'],
                results['train_mae_percentage'], results['test_mae_percentage']
            ]
        })
        stats_df.to_csv(stats_path, index=False)
        print(f"Model performance statistics saved to {stats_path}.")
    except Exception as e:
        print(f"Error saving model performance statistics to CSV: {e}")

    # Save feature importances to CSV
    try:
        varimp_path = r"C:\Users\jsten\PycharmProjects\PythonProject_test\models\feature_importances.csv"
        results['importance_df'].to_csv(varimp_path, index=False)
        print(f"Feature importances saved to {varimp_path}.")
    except Exception as e:
        print(f"Error saving feature importances to CSV: {e}")

    # Print results
    print("\n=== Final Results ===")
    print(f"Best Parameters: {results['grid_search'].best_params_}")
    print(f"Best CV MSE: {-results['grid_search'].best_score_:.4f}")
    print(f"Training MSE: {results['train_mse']:.4f}")
    print(f"Test MSE: {results['test_mse']:.4f}")
    print(f"Training MAE: {results['train_mae']:.4f} ({results['train_mae_percentage']:.2f}%)")
    print(f"Test MAE: {results['test_mae']:.4f} ({results['test_mae_percentage']:.2f}%)")
    print("\nFeature Importances:")
    print(results['importance_df'])

    # Plot CV results
    cv_results = -results['grid_search'].cv_results_['mean_test_score']
    plt.figure(figsize=(10, 6))
    plt.plot(cv_results)
    plt.title('Cross-Validation MSE Across Hyperparameter Combinations')
    plt.ylabel('MSE')
    plt.xlabel('Hyperparameter Combination Index')
    plt.show()