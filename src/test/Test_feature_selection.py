import unittest
import pandas as pd
from sklearn.datasets import make_regression
from model.feature_engineering.data_reduction import FeatureSelection

class TestFeatureSelection(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_regression(n_samples=100, n_features=5, random_state=42)

    def test_pca(self):
        feature_selector = FeatureSelection(pca=True, num_components=2)
        X_transformed = feature_selector.fit_transform(self.X)
        self.assertEqual(X_transformed.shape[1], 2)

    def test_k_best(self):
        feature_selector = FeatureSelection(k_best=True, k_features=2)
        X_transformed = feature_selector.fit_transform(self.X, self.y)
        self.assertEqual(X_transformed.shape[1], 2)

    def test_both_pca_and_k_best(self):
        feature_selector = FeatureSelection(pca=True, num_components=2, k_best=True, k_features=2)
        X_transformed = feature_selector.fit_transform(self.X, self.y)
        self.assertEqual(X_transformed.shape[1], 2)

if __name__ == '__main__':
    unittest.main()
