import unittest
import pandas as pd
import numpy as np
from regression_pipeline import UniversalRegressionPipeline
from sklearn.datasets import make_regression

class TestRegressionPipeline(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data
        X, y = make_regression(n_samples=200, n_features=5, noise=0.2, random_state=42)
        self.df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
        self.df['target'] = y
        self.df['cat_feat'] = np.random.choice(['X', 'Y', 'Z'], size=200)

        # Add some NaNs
        self.df.loc[0, 'feat_0'] = np.nan

    def test_pipeline_execution(self):
        pipeline = UniversalRegressionPipeline(target_column='target', cv=2)
        results = pipeline.fit(self.df)

        # Check if results dataframe is populated
        self.assertFalse(results.empty)
        self.assertIn('RMSE', results.columns)
        self.assertIn('R2', results.columns)
        self.assertIn('LinearRegression', results['Model'].values)
        self.assertIn('RandomForest', results['Model'].values)

    def test_prediction(self):
        pipeline = UniversalRegressionPipeline(target_column='target', cv=2)
        pipeline.fit(self.df)

        # Test prediction with best model
        preds = pipeline.predict(self.df.drop(columns=['target']))
        self.assertEqual(len(preds), len(self.df))

    def test_feature_detection(self):
        pipeline = UniversalRegressionPipeline(target_column='target', cv=2)
        pipeline.fit(self.df)

        self.assertIn('feat_0', pipeline.numeric_features)
        self.assertIn('cat_feat', pipeline.categorical_features)

if __name__ == '__main__':
    unittest.main()
