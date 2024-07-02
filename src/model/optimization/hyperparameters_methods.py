import optuna
import os
import shap
from sklearn.base import is_classifier
import logging
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, make_scorer,matthews_corrcoef


class Hyperparameters:
    """Hyperparameters: This class is used to tune the hyperparameters of a machine learning model using the Optuna library. It has a method called "get_models_params" that creates a voting classifier and the "call" method that performs cross-validation on the data to obtain a score, which is then returned."""
    def __init__(self, X, y):
        """
        Constructs all the necessary attributes for the Hyperparameters object.

        Parameters
        ----------
            X : Numeric or categorical data
            y : Numeric or categorical data
        """
        self.X = X
        self.y = y
        self.logger = logging.getLogger(__name__)

    def get_models_params(self, trial):
        """get_models_params: Function used to select and instantiate a machine learning classifier from a list of classifier options, including "RandomForestClassifier," "DecisionTreeClassifier," "ExtraTreesClassifier," and "SVC." The classifier selection is done using the "trial.suggest" function from the Optuna library. Finally, the code instantiates a "VotingClassifier" using the selected classifier and returns the VotingClassifier instance."""

        classifier_name = trial.suggest_categorical("classifier", ["RandomForestClassifier", "DecisionTreeClassifier", "ExtraTreesClassifier","KNN", "LightGBM", "CatBoost"])

        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        n_estimators = trial.suggest_int("n_estimators", 50, 150)
        max_depth = trial.suggest_int("max_depth", 2, 10)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 11)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 11)
        max_features = trial.suggest_categorical("max_features", [None])
        max_leaf_nodes = trial.suggest_categorical("max_leaf_nodes", [None])

        learning_rate = trial.suggest_float("learning_rate", 0.08, 0.5)
        num_leaves = trial.suggest_int("num_leaves", 100, 128)
        boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"])

        algorithm = trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
        leaf_size = trial.suggest_int("leaf_size", 10, 50)
        metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "chebyshev", "minkowski"])
        metric_params = trial.suggest_categorical("metric_params", [None])
        n_jobs = trial.suggest_categorical("n_jobs", [None])
        n_neighbors = trial.suggest_int("n_neighbors", 1, 20)
        p = trial.suggest_int("p", 1, 3)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])

        depth = trial.suggest_int("depth", 4, 10)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-5, 100)
        bagging_temperature = trial.suggest_float("bagging_temperature", 0, 1)
        border_count = trial.suggest_int("border_count", 1, 255)

        classifiers = {
            "RandomForestClassifier": RandomForestClassifier(
                criterion=criterion,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                n_jobs=-1,
                random_state=42
            ),
            "DecisionTreeClassifier": DecisionTreeClassifier(
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                random_state=42
            ),
            "ExtraTreesClassifier": ExtraTreesClassifier(
                criterion=criterion,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                random_state=42
            ),
            "KNN": KNeighborsClassifier(
                algorithm=algorithm,
                leaf_size=leaf_size,
                metric=metric,
                metric_params=metric_params,
                n_jobs=n_jobs,
                n_neighbors=n_neighbors,
                p=p,
                weights=weights
                
                                        
           ),
            "LightGBM": LGBMClassifier(
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                max_depth=max_depth,
                num_leaves=num_leaves,
                boosting_type=boosting_type,
                random_state=42
            ),
            "CatBoost": CatBoostClassifier(
                iterations=n_estimators,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                bagging_temperature=bagging_temperature,
                border_count=border_count,
                random_seed=42,
                logging_level='Silent'
        )
        }

        final_model = VotingClassifier(estimators=[(classifier_name, classifiers[classifier_name])], voting="soft")
        return final_model

    def __call__(self, trial):
        """The "call" method performs cross-validation using "cross_val_score" on the data to obtain a score, which is then returned."""
        final_model = self.get_models_params(trial)
        recall_scorer = make_scorer(recall_score)
        score = cross_val_score(final_model, self.X, self.y, n_jobs=-1, cv=5, scoring='accuracy').mean()
        trial.set_user_attr("final_model", final_model)
        return score


    def calculate_metrics_and_save_results(self, study, X_train_selected, X_test_selected, y_train, y_test):
        metrics = []
        for trial in study.trials:
            model = trial.user_attrs.get("final_model", None)
            if model:
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_test_selected)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                mcc = matthews_corrcoef(y_test, y_pred)


                metrics.append({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mcc": mcc
                })

        df_results = study.trials_dataframe()
        for idx, metric in enumerate(metrics):
            df_results.loc[idx, "accuracy"] = metric["accuracy"]
            df_results.loc[idx, "precision"] = metric["precision"]
            df_results.loc[idx, "recall"] = metric["recall"]
            df_results.loc[idx, "f1"] = metric["f1"]
            df_results.loc[idx, "mcc"] = metric["mcc"]
        select_results = df_results[[
            "params_classifier",
            "params_criterion",
            "params_n_estimators",
            "params_max_depth",
            "params_min_samples_split",
            "params_min_samples_leaf",
            "params_n_neighbors",
            "value",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
            "datetime_start",
            "datetime_complete",
            "duration",
        ]]

        results_folder = "results"
        df_results.to_csv(os.path.join(results_folder, "Hyperparameters_Results.csv"), index=False)
        ranking = select_results.sort_values("value", ascending=False).drop_duplicates(["params_classifier"]).head(5)
        ranking.to_csv(os.path.join(results_folder, "Models_Ranking.csv"), index=False)
        #self.logger.info(f"See the Models Ranking and Hyperparameters Results at: {results_folder}")
        formatted_ranking = "\n".join(
            f"Classifier: {row['params_classifier']}, Value: {row['value']:.4f}"
            for index, row in ranking.iterrows()
        )
        self.logger.info("Top ranked algorithms:\n%s", formatted_ranking)
        return formatted_ranking, df_results, select_results

