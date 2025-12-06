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
        Универсальный пайплайн для задач регрессии.

        Args:
            target_column (str): Название целевой переменной.
            numeric_features (list): Список названий численных столбцов. Если None, определяется автоматически.
            categorical_features (list): Список названий категориальных столбцов. Если None, определяется автоматически.
            models (dict): Словарь экземпляров моделей. Если None, используются модели по умолчанию.
            param_grids (dict): Словарь сеток параметров для GridSearchCV.
            scoring (str): Метрика оценки для оптимизации.
            cv (int): Количество фолдов кросс-валидации.
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

        # Модели по умолчанию, если не предоставлены
        if models is None:
            self.models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(random_state=random_state) if 'random_state' in Ridge().get_params() else Ridge(),
                'RandomForest': RandomForestRegressor(random_state=random_state),
                'GradientBoosting': GradientBoostingRegressor(random_state=random_state)
            }
        else:
            self.models = models

        # Сетки параметров по умолчанию, если не предоставлены
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
        Обучает пайплайн на датафрейме.
        """
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Автоматическое определение признаков, если они не указаны
        if self.numeric_features is None:
            self.numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        if self.categorical_features is None:
            self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        logger.info(f"Численные признаки: {self.numeric_features}")
        logger.info(f"Категориальные признаки: {self.categorical_features}")

        # Пайплайн предобработки
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

        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        results_list = []

        for name, model in self.models.items():
            logger.info(f"Обучение {name}...")

            # Создание полного пайплайна
            pipe = Pipeline(steps=[('preprocessor', self.preprocessor),
                                   ('regressor', model)])

            # Поиск по сетке (Grid Search)
            if name in self.param_grids and self.param_grids[name]:
                grid = GridSearchCV(pipe, self.param_grids[name], cv=self.cv, scoring=self.scoring, n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
                logger.info(f"Лучшие параметры для {name}: {best_params}")
            else:
                pipe.fit(X_train, y_train)
                best_model = pipe
                best_params = "По умолчанию"

            self.best_models[name] = best_model

            # Оценка качества
            y_pred = best_model.predict(X_test)
            metrics = self._evaluate(y_test, y_pred)
            metrics['Model'] = name
            metrics['Best Params'] = str(best_params)
            results_list.append(metrics)

        self.results = pd.DataFrame(results_list)
        logger.info("Обучение завершено.")
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
        Предсказание с использованием лучшей обученной модели.
        Если model_name равен None, используется лучшая модель на основе оценки R2 при обучении.
        """
        if not self.best_models:
            raise ValueError("Пайплайн не обучен. Сначала вызовите fit().")

        if model_name is None:
            # Выбор лучшей модели по R2
            best_model_name = self.results.loc[self.results['R2'].idxmax()]['Model']
            logger.info(f"Используется лучшая модель: {best_model_name}")
            model = self.best_models[best_model_name]
        else:
            if model_name not in self.best_models:
                raise ValueError(f"Модель {model_name} не найдена. Доступные: {list(self.best_models.keys())}")
            model = self.best_models[model_name]

        # Убедимся, что у нас есть только столбцы признаков (если целевая переменная присутствует, но игнорируется)
        if self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
        else:
            X = df

        return model.predict(X)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Пример использования
    from sklearn.datasets import make_regression

    # Генерация синтетических данных
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    df['target'] = y

    # Введение пропущенных значений и категориальных признаков для тестирования
    df['category'] = np.random.choice(['A', 'B', 'C'], size=len(df))
    df.loc[0:10, 'feature_0'] = np.nan

    pipeline = UniversalRegressionPipeline(target_column='target')
    results = pipeline.fit(df)
    print("\nРезультаты:")
    print(results)
