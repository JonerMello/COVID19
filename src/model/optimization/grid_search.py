from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
class GridSearch:
    """Hiperparâmetros: esta classe é usada para ajustar os hiperparâmetros de um modelo de aprendizado de máquina usando a biblioteca Grid Search. Possui um método chamado " get_models_params " que cria um classificador de votação e o método " call " que realiza validação cruzada nos dados para obter uma pontuação, que é então retornada."""
    def __init__(self, X, y):
        """
        Constrói todos os atributos necessários para o objeto Hyperparameters.

        Parameters
        ----------
            X : Dados numéricos ou de categoricos
            y : Dados numéricos ou de categoricos
        """
        self.X = X
        self.y = y

    def optimize(self):
        models = [
            {
                'name': 'Random Forest Classifier',
                'model': RandomForestClassifier(),
                'params': {
                    'n_estimators': [10, 50, 100],
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            {
                'name': 'SVC',
                'model': SVC(),
                'params': {
                    'C': [1, 10, 100],
                    'kernel': ['linear', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto']
                }
            },
            {
                'name': 'Decision Tree Classifier',
                'model': DecisionTreeClassifier(),
                'params': {
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            {
                'name': 'AdaBoost Classifier',
                'model': AdaBoostClassifier(),
                'params': {
                    'n_estimators': [10, 50, 100],
                    'learning_rate': [0.1, 1, 10]
                }
            },
            {
                'name': 'Extra Trees Classifier',
                'model': ExtraTreesClassifier(),
                'params': {
                    'n_estimators': [10, 50, 100],
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            }
        ]
        
        best_score = 0
        best_model = None
        
        for model in models:
            print("model__",model)
            clf = GridSearchCV(model['model'], model['params'], cv=5)
            clf.fit(self.X, self.y)
            print("OKOKOKOK",clf)
            if clf.best_score_ > best_score:
                best_score = clf.best_score_
                best_model = model['name']
                print("best_score",best_score)
                print("best_model",best_model)
        
        print(f'Best score: {best_score:.4f} using {best_model}')
        return best_model,best_score