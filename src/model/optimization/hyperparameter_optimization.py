
from sklearn.base import BaseEstimator, TransformerMixin
from model.optimization.hyperparameters_methods import Hyperparameters
class HyperparameterOptimization(BaseEstimator, TransformerMixin):
    def __init__(self, X, y, n_trials):
        self.X = X
        self.y = y
        self.n_trials = n_trials

    def fit(self, X, y=None):
        # Perform hyperparameter optimization
        self.hyperparameter_optimizer = Hyperparameters(self.X, self.y)
        self.study = self.hyperparameter_optimizer.run_optimization(n_trials=self.n_trials)
        self.best_model = self.study.best_user_attrs["final_model"]
        return self

    def transform(self, X):
        # You can return X as is or make any necessary transformations
        return X