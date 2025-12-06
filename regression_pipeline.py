import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Configure logging
logger = logging.getLogger(__name__)

class UniversalRegressionPipeline:
    def __init__(self, target_column, numeric_features=None, categorical_features=None,
                 models=None, param_grids=None, scoring='neg_mean_squared_error', cv=5, random_state=42):
        """
        Universal Pipeline for Regression Tasks.

        Args:
            target_column (str): Name of the target variable.
            numeric_features (list): List of numerical column names. If None, auto-detected.
            categorical_features (list): List of categorical column names. If None, auto-detected.
            models (dict): Dictionary of model instances. If None, defaults are used.
            param_grids (dict): Dictionary of parameter grids for GridSearchCV.
            scoring (str): Scoring metric for optimization.
            cv (int): Number of cross-validation folds.
            random_state (int): Random seed.
        """
        self.target_column = target_column
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.best_models = {}
        self.results = {}

        # Default models if not provided
        if models is None:
            self.models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(random_state=random_state) if 'random_state' in Ridge().get_params() else Ridge(),
                'RandomForest': RandomForestRegressor(random_state=random_state),
                'GradientBoosting': GradientBoostingRegressor(random_state=random_state)
            }
        else:
            self.models = models

        # Default parameter grids if not provided
        if param_grids is None:
            self.param_grids = {
                'LinearRegression': {},
                'Ridge': {'regressor__alpha': [0.1, 1.0, 10.0]},
                'RandomForest': {
                    'regressor__n_estimators': [50, 100, 200],
                    'regressor__max_depth': [None, 10, 20],
                    'regressor__min_samples_split': [2, 5]
                },
                'GradientBoosting': {
                    'regressor__n_estimators': [50, 100],
                    'regressor__learning_rate': [0.01, 0.1, 0.2],
                    'regressor__max_depth': [3, 5]
                }
            }
        else:
            self.param_grids = param_grids

    def fit(self, df):
        """
        Fits the pipeline to the dataframe.
        """
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Auto-detect features if not provided
        if self.numeric_features is None:
            self.numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if self.categorical_features is None:
            self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        logger.info(f"Numeric features: {self.numeric_features}")
        logger.info(f"Categorical features: {self.categorical_features}")

        # Preprocessing Pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        results_list = []

        for name, model in self.models.items():
            logger.info(f"Training {name}...")

            # Create full pipeline
            pipe = Pipeline(steps=[('preprocessor', self.preprocessor),
                                   ('regressor', model)])

            # Grid Search
            if name in self.param_grids and self.param_grids[name]:
                grid = GridSearchCV(pipe, self.param_grids[name], cv=self.cv, scoring=self.scoring, n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
                logger.info(f"Best params for {name}: {best_params}")
            else:
                pipe.fit(X_train, y_train)
                best_model = pipe
                best_params = "Default"

            self.best_models[name] = best_model

            # Evaluate
            y_pred = best_model.predict(X_test)
            metrics = self._evaluate(y_test, y_pred)
            metrics['Model'] = name
            metrics['Best Params'] = str(best_params)
            results_list.append(metrics)

        self.results = pd.DataFrame(results_list)
        logger.info("Training complete.")
        return self.results

    def _evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

    def predict(self, df, model_name=None):
        """
        Predict using the best trained model.
        If model_name is None, uses the best model based on R2 score from training.
        """
        if not self.best_models:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        if model_name is None:
            # Select best model based on R2
            best_model_name = self.results.loc[self.results['R2'].idxmax()]['Model']
            logger.info(f"Using best model: {best_model_name}")
            model = self.best_models[best_model_name]
        else:
            if model_name not in self.best_models:
                raise ValueError(f"Model {model_name} not found. Available: {list(self.best_models.keys())}")
            model = self.best_models[model_name]

        # Ensure we only have feature columns (in case target is present but ignored, usually predict expects X)
        if self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
        else:
            X = df

        return model.predict(X)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Example Usage
    from sklearn.datasets import make_regression

    # Generate synthetic data
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    df['target'] = y

    # Introduce some missing values and categorical features for testing
    df['category'] = np.random.choice(['A', 'B', 'C'], size=len(df))
    df.loc[0:10, 'feature_0'] = np.nan

    pipeline = UniversalRegressionPipeline(target_column='target')
    results = pipeline.fit(df)
    print("\nResults:")
    print(results)
