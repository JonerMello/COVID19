import unittest
import optuna
from sklearn.datasets import make_classification
from model.optimization.hyperparameters_methods import Hyperparameters
from sklearn.ensemble import VotingClassifier

class TestHyperparameters(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(n_samples=100, n_features=5, random_state=42)
        self.hyperparameters = Hyperparameters(self.X, self.y)

    def test_get_models_params(self):
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        model = self.hyperparameters.get_models_params(trial)
        self.assertIsInstance(model, VotingClassifier)

    def test_call(self):
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        score = self.hyperparameters(trial)
        self.assertGreaterEqual(score, 0.0)

if __name__ == '__main__':
    unittest.main()
