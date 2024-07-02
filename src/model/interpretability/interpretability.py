import os
import datetime
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.base import is_classifier
import shap
from lime.lime_tabular import LimeTabularExplainer
import joblib
from collections import defaultdict
from model.optimization.hyperparameters_methods import Hyperparameters

class Interpretability:
    def __init__(self, best_model, X_train_selected, X_test_selected, y_train, transformed_feature_names_df, results_folder, use_pca=False, use_anova=False, use_lasso=False):
        self.best_model = best_model
        self.X_train_selected = X_train_selected
        self.X_test_selected = X_test_selected
        self.y_train = y_train
        self.transformed_feature_names_df = transformed_feature_names_df
        self.results_folder = results_folder
        self.use_pca = use_pca
        self.use_anova = use_anova
        self.use_lasso = use_lasso
        self._explainers_cache = {}  # Inicialização do cache de Explainers

    def explain_model(self):
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.use_pca:
            # Escolha uma amostra (índice de amostra) do conjunto de teste que deseja explicar
            sample_idx = 0  # Substitua pelo índice da amostra que deseja explicar

            for name, estimator in self.best_model.named_estimators_.items():
                if isinstance(estimator, LGBMClassifier):
                    return None, None, None
                if isinstance(estimator, (RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier)):
                    background_data = shap.sample(self.X_train_selected, 1000)
                    explainer = shap.TreeExplainer(estimator, background_data=background_data)
                elif isinstance(estimator, KNeighborsClassifier):
                    background_data_kmeans = shap.kmeans(self.X_train_selected, 100)
                    explainer = shap.KernelExplainer(estimator.predict_proba, background_data_kmeans)

                if isinstance(estimator, KNeighborsClassifier):
                    shap_values = explainer.shap_values(self.X_test_selected[sample_idx], nsamples=20)[0]
                else:
                    shap_values = explainer.shap_values(self.X_test_selected[sample_idx])[0]

                shap.initjs()
                shap.force_plot(explainer.expected_value[0], shap_values, feature_names=self.transformed_feature_names_df.columns, matplotlib=True, show=False)

                shap_force_filename = f'interpretability_{current_datetime}_{name}.html'
                shap_exp_filepath = os.path.join(self.results_folder, shap_force_filename)
                shap.save_html(shap_exp_filepath, shap.force_plot(explainer.expected_value[0], shap_values, feature_names=self.transformed_feature_names_df.columns))

                explainer = LimeTabularExplainer(self.X_train_selected, mode="classification", training_labels=self.y_train, feature_names=self.transformed_feature_names_df)

                explanation = explainer.explain_instance(self.X_test_selected[sample_idx], self.best_model.predict_proba)
                exp_filename = f'interpretability_{current_datetime}.html'
                exp_filepath = os.path.join(self.results_folder, exp_filename)
                explanation.save_to_file(exp_filepath)

                feature_importances = explanation.as_list()
                feature_names = [feature for feature, _ in feature_importances]
                importances = [importance for _, importance in feature_importances]

                plt.figure(figsize=(12, len(feature_names) * 0.5))
                plt.barh(feature_names, importances)
                plt.xlabel('Importância')
                plt.ylabel('Feature')
                plt.title('Importância das Features')
                plt.tight_layout()
                plt.margins(y=0.01)

                exp_filename = f'lime_feature_importance_{current_datetime}.jpg'
                lime_exp_filepath = os.path.join(self.results_folder, exp_filename)
                plt.savefig(lime_exp_filepath, format='jpg', dpi=300)
                plt.close()

                return shap_exp_filepath, exp_filepath, lime_exp_filepath
        else:
            print("Nenhum método de seleção de recursos habilitado")

        
        if self.use_anova:
            return None, None, None
        if not self.use_anova or not self.use_pca:
            return None, None, None




    @staticmethod
    def explanation(best_trial, X_train, y_train, X_test,feature_names):
        final_model = Hyperparameters(X_train, y_train).get_models_params(best_trial)
        final_model.fit(X_train, y_train)
        
        shap_values = None
        plot_type = None
        feature_names_clean = [name.replace('_1.0', '') for name in feature_names]
 
        for name, estimator in final_model.named_estimators_.items():
            if isinstance(estimator, KNeighborsClassifier):
                background = shap.sample(X_train, 100)
                explainer = shap.KernelExplainer(estimator.predict, background, feature_names=feature_names_clean)
                shap_values = explainer.shap_values(X_test)
                plot_type = "beeswarm"
            elif isinstance(estimator, (RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier, LGBMClassifier)):
                explainer = shap.TreeExplainer(estimator, X_train, feature_names=feature_names_clean ,check_additivity=False)
                shap_values = explainer.shap_values(X_test)
                plot_type = "summary"
            elif isinstance(estimator, CatBoostClassifier):
                explainer = shap.TreeExplainer(estimator, X_train, feature_names=feature_names_clean, check_additivity=False)
                shap_values = explainer(X_test)
                plot_type = "summary"
            elif is_classifier(estimator):
                explainer = shap.Explainer(estimator, X_train, feature_names=feature_names, check_additivity=False)
                shap_values = explainer(X_test)
                plot_type = "summary"
            else:
                continue
            
            if shap_values is not None:
                break

        if shap_values is None:
            raise ValueError("No callable estimator found within the VotingClassifier.")
        
        plt.figure()
    
        if plot_type == "beeswarm":
            fig_explanation = shap.plots.beeswarm(shap_values, max_display=20, show=False)
        elif plot_type == "summary":
            fig_explanation = shap.summary_plot(shap_values, X_test, show=False)
        elif plot_type == "force":
            explainer = shap.Explainer(shap_values)
            shap.plots.force(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
        elif plot_type == "decision":
            explainer = shap.Explainer(shap_values)
            shap.plots.decision(explainer.expected_value, shap_values[0,:])
        else:
            raise ValueError("Unsupported plot_type. Use 'beeswarm', 'summary', 'force', or 'decision'.")
    
        plt.savefig(f"shap_{plot_type}_plot.png")
        plt.close()
        
        return shap_values

    @staticmethod
    def lime_explanation(best_trial, X_train, y_train, X_test, feature_names):
        final_model = Hyperparameters(X_train, y_train).get_models_params(best_trial)
        final_model.fit(X_train, y_train)

        lime_explanation = None

        # Remove sufixos "_1.0" dos nomes das features
        cleaned_feature_names = [name.replace('_1.0', '') for name in feature_names]

        explainer = LimeTabularExplainer(X_train, mode="classification", training_labels=y_train, feature_names=cleaned_feature_names)
        sample_idx = 0  # Escolha o índice da amostra que deseja explicar
        lime_explanation = explainer.explain_instance(X_test[sample_idx], final_model.predict_proba)

        # Salvar a explicação em um arquivo HTML
        exp_filename = f'interpretability_{sample_idx}.html'
  
        lime_explanation.save_to_file(exp_filename)

        # Plotar a importância das features
        feature_importances = lime_explanation.as_list()
        feature_names = [feature for feature, _ in feature_importances]
        importances = [importance for _, importance in feature_importances]

        plt.figure(figsize=(12, len(feature_names) * 0.5))
        plt.barh(feature_names, importances)
        plt.xlabel('Importância')
        plt.ylabel('Feature')
        plt.title('Importância das Features')
        plt.tight_layout()
        plt.margins(y=0.01)

        exp_filename = f'lime_feature_importance.jpg'
        plt.savefig(exp_filename, format='jpg', dpi=300)
        plt.close()