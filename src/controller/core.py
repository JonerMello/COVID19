import os
import pandas as pd
import numpy as np
import csv
import logging
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import time
import mlflow
import threading
import webbrowser
from mlflow import MlflowClient
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset
import shap
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import subprocess
import base64
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import plotly.io as pio
import plotly.graph_objs as go
import plotly.offline as pyo
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,mean_squared_error,matthews_corrcoef
from halo import Halo
from model.tools.dataset_validation import DatasetValidation
from model.preprocessing.data_cleaning import DataCleaning
from model.preprocessing.data_info import DataInfo
from model.preprocessing.data_transformation import DataTransformation
from model.feature_engineering.data_reduction import FeatureSelection
from model.optimization.hyperparameters_methods import Hyperparameters
from model.interpretability.interpretability import Interpretability
import io
from sklearn.pipeline import FeatureUnion
from model.optimization.grid_search import GridSearch
from sklearn.pipeline import make_pipeline
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
import joblib
import datetime
from sklearn.utils import estimator_html_repr
import codecs
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.backends.backend_agg import FigureCanvasAgg
from colorama import Fore, Style
import warnings
from mlflow.models import infer_signature
import psutil
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor


from optuna.exceptions import ExperimentalWarning
class Core:
    # Desativar todos os UserWarnings temporariamente
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    mlflow.set_system_metrics_sampling_interval(1)
    def __init__(self, dataset_url, label, log_level=logging.INFO, remove_duplicates=True,
                 remove_missing_values=True, remove_outliers=True, one_hot_encoder=True,
                 do_label_encode=True, balance_classes=True):
        self.dataset_url = dataset_url
        self.label = label
        self.use_pca = False 
        self.use_anova = False
        self.use_lasso =False
        self.log_level = log_level
        self.remove_duplicates = remove_duplicates
        self.remove_missing_values = remove_missing_values
        self.remove_outliers = remove_outliers
        self.one_hot_encoder = one_hot_encoder
        self.do_label_encode = do_label_encode
        self.balance_classes = balance_classes
        self.measurements = []

        # Mapear nomes dos níveis de log para valores inteiros
        log_level_mapping = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        
        # Verificar se o log_level fornecido é válido
        if log_level not in log_level_mapping:
            raise ValueError(f"Invalid log level: {log_level}")
        
        # Configure logging settings
        logging.basicConfig(level=log_level_mapping[log_level], format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def create_results_folder(self, folder_name="results"):
        """
        Create a folder with the specified name if it doesn't exist.

        Args:
            folder_name (str): The name of the folder to create.

        Returns:
            str: The path to the created folder.
        """
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        return folder_name 

    def plot_train_test_relation(self, y_train, y_test):
        # Mapping labels to classes
        class_labels = {
            0: 'Benign',
            1: 'Malware'
            # Add more labels for other classes as needed
        }

        # Count classes in training and test sets
        train_counts = np.bincount(y_train)
        test_counts = np.bincount(y_test)

        # Get the maximum number of classes between training and test to ensure both have the same categories in the plot
        max_classes = max(len(train_counts), len(test_counts))

        # Pad with zeros to ensure both have the same number of categories
        train_counts = np.pad(train_counts, (0, max_classes - len(train_counts)), mode='constant')
        test_counts = np.pad(test_counts, (0, max_classes - len(test_counts)), mode='constant')

        # Create a stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        index = np.arange(max_classes)
        bar_width = 0.35

        bars1 = ax.bar(index, train_counts, bar_width, label='Training Set')
        bars2 = ax.bar(index, test_counts, bar_width, bottom=train_counts, label='Test Set')

        # Add labels to the bars
        for i, label in class_labels.items():
             ax.text(i, train_counts[i] / 2, f"{train_counts[i]}", ha='center', va='center_baseline', color='white', fontweight='bold')
             ax.text(i, train_counts[i] + test_counts[i] / 2, f"{test_counts[i]}", ha='center', va='center_baseline', color='white', fontweight='bold')

        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Classes in Training and Test Sets')
        ax.set_xticks(index)
        ax.set_xticklabels([class_labels.get(i, '') for i in range(max_classes)])  # Set class labels on x-axis
        ax.legend()

        # Save the plot as a PNG image
        results_folder = "results"
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        img_filename = f'train_test_distribution.png'
        img_filepath = os.path.join(results_folder, img_filename)
        plt.savefig(img_filepath, bbox_inches='tight')
        plt.close()
    
    def display_data_info(self, dataset_df):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        colored_message = f"[{Fore.GREEN}{now}{Style.RESET_ALL}] Displaying data info..."
        logging.info(colored_message)
        data_info = DataInfo(self.label, dataset_df)
        data_info.display_dataframe_info()
        return data_info

    def preprocess(self, dataset_df, label):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        colored_message = f"[{Fore.GREEN}{now}{Style.RESET_ALL}] Preprocessing data..."
        logging.info(colored_message)

        # Check the data types of each column in the DataFrame
        data_types = dataset_df.dtypes

        # Check if dataset_df contains numerical columns
        contains_numerical = data_types != 'int64'

        if contains_numerical.any():
            one_hot_encoder = True
        else:
            one_hot_encoder = False

        preprocessor = Pipeline(steps=[
            ('Data Cleaning', DataCleaning(remove_duplicates=self.remove_duplicates, remove_missing_values=self.remove_missing_values, remove_outliers=self.remove_outliers)),
            ('Data Transformation', DataTransformation(label=self.label, one_hot_encoder=one_hot_encoder, do_label_encode=self.do_label_encode))
        ])
        
        # Fit and transform your dataset using the pipeline
        transformation = preprocessor.fit_transform(dataset_df)
        X, y = transformation[0].astype(np.int8), transformation[1].astype(np.int8)
        
        return X, y, preprocessor
  

    def enable_pca(self):
        self.use_pca = True

    def enable_anova(self):
        self.use_anova = True

    def enable_lasso(self):
        self.use_lasso = True
    

    def feature_selection(self, X, y):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        colored_message = f"[{Fore.GREEN}{now}{Style.RESET_ALL}] Performing feature selection"
        logging.info(colored_message)
        # Save the DataFrames to CSV in the results folder
        results_folder = self.create_results_folder()
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.plot_train_test_relation(y_train, y_test)
        feature_selection_steps = []
        if self.balance_classes == True:
            feature_selection_steps.append(('balancing', FeatureSelection(balance_classes=False)))
        if self.use_pca:
            feature_selection_steps.append(('PCA', FeatureSelection(pca=True, num_components=int(0.3 * X.shape[1]))))
            # Aplicar PCA para redução de dimensionalidade com 2 componentes

        if self.use_anova:
            feature_selection_steps.append(('ANOVA', FeatureSelection(anova=True, k_features=int(0.3 * X.shape[1]))))

        if self.use_lasso:
            feature_selection_steps.append(('LASSO', FeatureSelection(lasso=True, alpha=0.00001)))

        if not self.use_pca and not self.use_anova and not self.use_lasso:
            colored_message_not_feature = f"[{Fore.YELLOW}No feature selection method enabled. Using original features.{Style.RESET_ALL}]"
            self.logger.warning(colored_message_not_feature)

        

        feature_selection_pipeline = Pipeline(feature_selection_steps) if feature_selection_steps else None
        transformed_feature_names_df = None

        # Fit and transform with the selected feature selection methods
        if feature_selection_pipeline:
            X_train_selected = feature_selection_pipeline.fit_transform(X_train, y_train)
            X_test_selected = feature_selection_pipeline.transform(X_test)
            
            # Recupere os nomes das colunas transformadas
            transformed_feature_names = []

            for step_name, step_obj in feature_selection_pipeline.named_steps.items():
                if isinstance(step_obj, FeatureSelection):
                    if step_obj.feature_names:
                        transformed_feature_names.extend([f"{name}" for name in step_obj.feature_names])
                    else:
                        # Obtenha o número real de recursos selecionados
                        num_selected_features = X_train_selected.shape[1]
                        # Obtenha apenas os primeiros 'num_selected_features' nomes de colunas transformadas
                        transformed_feature_names.extend([f"{name}" for i in range(num_selected_features)])

            # Certifique-se de que 'transformed_feature_names' não tenha mais nomes de colunas do que as colunas reais
            transformed_feature_names = transformed_feature_names[:X_train_selected.shape[1]]

            # Crie um DataFrame com base nos nomes das colunas transformadas
            transformed_feature_names_df = pd.DataFrame(X_train_selected, columns=transformed_feature_names)
            current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            feature_selection_filename = f'Features_Selected_{current_datetime}.csv'
            features_selected = os.path.join(results_folder, feature_selection_filename)
            transformed_feature_names_df.to_csv(features_selected, index=False)
            self.logger.info(f"See the selected features at: {features_selected}")
            self.logger.debug(transformed_feature_names_df)

           
        else:
            X_train_selected = X_train
            X_test_selected = X_test
        
        return X_train_selected, X_test_selected, y_train, y_test, feature_selection_pipeline,transformed_feature_names_df

    def early_stopping_callback(self, study, trial):
        if trial.number > 5 and study.best_value > 0.98:
            study.stop()

    def optimize_hyperparameters(self, X, y):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        colored_message = f"[{Fore.GREEN}{now}{Style.RESET_ALL}] Optimizing hyperparameters..."
        logging.info(colored_message)
       
        hyperparameters = Hyperparameters(X, y)
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
        optuna.logging.disable_default_handler()
        study = optuna.create_study(study_name="distributed-study",direction="maximize", pruner=optuna.pruners.MedianPruner())
        study.optimize(hyperparameters,timeout=1200, gc_after_trial=True, n_trials=20, show_progress_bar=True,catch=(ValueError,), callbacks=[self.early_stopping_callback])

        

        # Step 3.2: Obtain the best model from hyperparameter optimization
        best_model = study.best_trial.user_attrs.get("final_model", None)

        return study,best_model

    def evaluate_model(self, model, X_test, y_test):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        colored_message = f"[{Fore.GREEN}{now}{Style.RESET_ALL}] Evaluating model..."
        logging.info(colored_message)
        
        y_pred = model.predict(X_test)
     
        report = classification_report(y_test, y_pred, target_names=["Benign", "Malware"], output_dict=True)

        self.logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))
        return report, y_test, y_pred

    def start_mlflow_ui(self):
        # Usando subprocess para iniciar o servidor MLflow de forma independente
        subprocess.Popen(["mlflow", "ui"])

    def open_browser(self):
        time.sleep(5)  # Espera 5 segundos para o servidor iniciar
        webbrowser.open_new_tab("http://localhost:5000")

    def measure_performance(self):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # em MB
        return start_time, start_memory

    def save_performance(self, step_name, start_time, start_memory):
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # em MB
        elapsed_time = end_time - start_time
        memory_usage = end_memory - start_memory

        self.measurements.append({
            'Step Name': step_name,
            'Elapsed Time (seconds)': f"{elapsed_time:.2f}",
            'Memory Usage (MB)': f"{memory_usage:.2f}"
        })

    def export_to_csv(self, filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Step Name', 'Elapsed Time (seconds)', 'Memory Usage (MB)'])
            writer.writeheader()
            for measurement in self.measurements:
                writer.writerow(measurement)

    def plot_performance(self, filename='performance_metrics.jpg'):
        step_names = [m['Step Name'] for m in self.measurements]
        elapsed_times = [float(m['Elapsed Time (seconds)']) for m in self.measurements]
        memory_usages = [float(m['Memory Usage (MB)']) for m in self.measurements]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_title('Performance Metrics per Step')
        ax1.set_xlabel('Step Name')
        ax1.set_ylabel('Elapsed Time (seconds)', color='tab:blue')
        bars = ax1.bar(step_names, elapsed_times, color='tab:blue', alpha=0.6, label='Elapsed Time (seconds)')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        for bar in bars:
            yval = bar.get_height()
            ax1.annotate(f'{yval:.2f}', 
                         xy=(bar.get_x() + bar.get_width() / 2, yval), 
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords='offset points', 
                         ha='center', va='bottom', fontsize=10, color='blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Memory Usage (MB)', color='tab:green')
        line = ax2.plot(step_names, memory_usages, color='tab:green', marker='o', label='Memory Usage (MB)')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        for i, txt in enumerate(memory_usages):
            ax2.annotate(f'{txt:.2f}', 
                         xy=(step_names[i], memory_usages[i]), 
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords='offset points', 
                         ha='center', va='bottom', fontsize=10, color='green')

        # Adicionar a legenda
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        fig.tight_layout()

        # Salvar o gráfico em um arquivo JPG
        plt.savefig(filename, format='jpg')

        # Fechar a figura para liberar memória
        plt.close()


    # Função para logar artefatos se o arquivo existir e não for None
    def log_artifact_if_exists(self,file_path, artifact_path):
        if file_path and os.path.exists(file_path):
            mlflow.log_artifact(file_path, artifact_path)


    def run(self):
        validator = DatasetValidation(self.dataset_url, self.label)
        results_folder= self.create_results_folder()
        
        if not validator.validate_dataset():
            return None 

        # Iniciando o MLflow UI em segundo plano
        self.start_mlflow_ui()
        self.open_browser()
        mlflow.set_experiment("MH-AutoML")

        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f'report_{current_datetime}.pdf'
        
        # Step 1: Data info
        start_time_step1, start_memory_step1 = self.measure_performance()
        dataset_df = validator.load_data()   
        display_data = self.display_data_info(dataset_df)
        self.save_performance('Data info', start_time_step1, start_memory_step1)

      

        # Step 2: Preprocessing
        start_time_step2, start_memory_step2 = self.measure_performance()
        X, y, preprocessor = self.preprocess(dataset_df, self.label)
        self.save_performance('Preprocessing', start_time_step2, start_memory_step2)


        # Step 3: Dimensionality Reduction
        start_time_step3, start_memory_step3 = self.measure_performance()
        if dataset_df.shape[1] > 50:
            #self.enable_anova()
            #self.enable_pca()
            self.enable_lasso()

        else:
            colored_message_not_features = f"[{Fore.YELLOW}The number of features in the dataset is less than or equal to 50, there is no need to reduce it.{Style.RESET_ALL}]"
            self.logger.warning(colored_message_not_features)
           

        X_train_selected, X_test_selected, y_train, y_test, selection_pipeline,transformed_feature_names_df = self.feature_selection(X, y)
        self.save_performance('Feature Eng', start_time_step3, start_memory_step3)
        

        # Step 4: Hyperparameter Optimization
        start_time_step4, start_memory_step4 = self.measure_performance()
        hyperparameters = Hyperparameters(X_train_selected, y_train)
        study, best_model = self.optimize_hyperparameters(X_train_selected, y_train)   

        formatted_ranking, df_results, select_results = hyperparameters.calculate_metrics_and_save_results(study, X_train_selected, X_test_selected, y_train, y_test)
        steps = [('preprocessor', preprocessor),("reduce_dim",selection_pipeline), ("classifier", best_model)]
        pipeline = Pipeline(steps)
        model_name = best_model.estimators[0][0]
        model_instance = best_model.estimators[0][1]
        model_params = model_instance.get_params()

        self.logger.info(f"Best Model: {model_name}, Best Parameters: {model_params}")
        # Fit the best model with the training data using X_train_pca and y_train
        best_model.fit(X_train_selected, y_train)
        y_pred = best_model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        self.save_performance('Hyperparameter', start_time_step4, start_memory_step4)

        # Step 5: Interpretability:
        start_time_step5, start_memory_step5 = self.measure_performance()
        interpretability = Interpretability(best_model, X_train_selected, X_test_selected, y, transformed_feature_names_df, results_folder, self.use_pca, self.use_anova, self.use_lasso)
        shap_exp_filepath, exp_filepath, lime_exp_filepath = interpretability.explain_model()

       
        if self.use_lasso:

            shap_values = Interpretability.explanation(study.best_trial, X_train_selected, y_train, X_test_selected,transformed_feature_names_df.columns)

            lime_explanation = interpretability.lime_explanation(study.best_trial, X_train_selected, y_train, X_test_selected, transformed_feature_names_df.columns)

        self.save_performance('Interpretability', start_time_step5, start_memory_step5)
        # Step 6: Evaluation and report
        start_time_step6, start_memory_step6 = self.measure_performance()
     
        # Concatenar os dados de treino e a variável alvo em um DataFrame do Pandas
        df = pd.DataFrame(data=X_train_selected, columns=[f'{transformed_feature_names_df}' for i in range(X_train_selected.shape[1])])
        df['class'] = y_train
    
        # Salvar o DataFrame em um arquivo CSV
        treino_filename = f'treino_{current_datetime}.csv'
        filepath_treino = os.path.join(results_folder, treino_filename)
        df.to_csv(filepath_treino, index=False)
        #print(f'Dados de treino salvos em {filepath_treino}')

        #######################################################################################

        model_filename = f'best_model_{current_datetime}.pkl'
        model_filepath = os.path.join(results_folder, model_filename)
        with open(model_filepath, 'wb') as model_file:
            pickle.dump(best_model, model_file)

        print("\n")
        self.logger.info(f"See your Best Model at: {model_filepath}")
        report, y_test, y_pred = self.evaluate_model(best_model, X_test_selected, y_test)
        print("\n")

        
        self.logger.info("Pipeline Configs")
        self.logger.info(pipeline)

        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f'report_{current_datetime}.html'
        # Concatenando o horário atual com a URL do conjunto de dados
        run_name = f"{current_datetime}_{self.dataset_url}"

            

        # Create an HTML file
        with open(report_filename, 'w', encoding='utf-8') as report_file:
            # Write the HTML header
            report_file.write("<html><head><title>MH-AutoML Report</title></head><body>")

            # Pipeline Configurations Section ###################################################################
            report_file.write("<h1>Pipeline Configurations</h1>")
            set_config(display="diagram")
            output_html = estimator_html_repr(pipeline)

            # Centralizar o conteúdo na página
            report_file.write('<div style="text-align: center;">')
            report_file.write(output_html)
            report_file.write('</div>')
            #####################################################################################################
              
            # Data Information Section #########################################################################          
            system_info = display_data.system_info_result.to_html(index=False)
            info_table = display_data.info_table_result.to_html(index=False)

            data_types = display_data.data_types_result.to_html(index=False)
            balance_info = display_data.balance_info_result.to_html(index=False)
            duplicates_missing = display_data.duplicates_missing_result.to_html(index=False)
            features_info = display_data.features_info_result.to_html(index=False)

            # Create the CSS styles
            table_style = """
            <style>
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                th, td {
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                }
            </style>
            """

            # Create the HTML content with the CSS styles
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Data Info</title>
                {table_style}
            </head>
            <body>
                <h1>System Information</h1>
                <table>
                {system_info}
                </table>
                <h1>Data Information</h1>
                <table>
                {info_table}
                </table>
                <h1>Data Type</h1>
                <table>
                {data_types}
                </table>
                <h1>Data Balancing</h1>
                <table>
                {balance_info}
                </table>
                <h1>Data Small</h1>
                <table>
                {duplicates_missing}
                </table>
                <h1>Features Info</h1>
                <table>
                {features_info}
                </table>
            </body>
            </html>
            """

            # Write the HTML content to the report file
            report_file.write(html_content)

            # Feature Selection Section
            if self.use_pca:
                # Feature Selection Section
                report_file.write("<h1>Feature Selection</h1>")

                # Get only the column names from the DataFrame
                feature_names = transformed_feature_names_df.columns
                feature_names = [name.rstrip('_1') for name in feature_names]
                # Create an HTML table and apply CSS styles
                report_file.write("<table style='border-collapse: collapse; width: 50%;'>")
                report_file.write("<tr style='background-color: #f2f2f2;'>")
                report_file.write("<th style='padding: 8px; border: 1px solid #dddddd; text-align: left;'>Column Name</th>")
                report_file.write("</tr>")

                for name in feature_names:
                    report_file.write("<tr>")
                    report_file.write(f"<td style='padding: 8px; border: 1px solid #dddddd;'>{name}</td>")
                    report_file.write("</tr>")

                report_file.write("</table>")

            #######################################################################################################
            # Hyperparameter Optimization Results Section #########################################################
      
            report_file.write("<h1>Ranking Hyperparameter Optimization</h1>")
            # Sort the results table by the "accuracy" column in descending order
            select_results = select_results.sort_values(by=['value', 'accuracy'], ascending=False)
            select_results.drop_duplicates(["params_classifier"])
            # Create an HTML table and apply CSS styles
            report_file.write("<table style='border-collapse: collapse; width: auto;' class='styled-table'>")
            report_file.write("<tr style='background-color: #f2f2f2;'>")

            report_file.write("<th style='padding: 8px; border: 1px solid #dddddd; text-align: left;'>Model</th>")
            report_file.write("<th style='padding: 8px; border: 1px solid #dddddd; text-align: left;'>Criterion</th>")
            report_file.write("<th style='padding: 8px; border: 1px solid #dddddd; text-align: left;'>n_estimators</th>")
            report_file.write("<th style='padding: 8px; border: 1px solid #dddddd; text-align: left;'>max_depth</th>")
            report_file.write("<th style='padding: 8px; border: 1px solid #dddddd; text-align: left;'>min_samples_split</th>")
            report_file.write("<th style='padding: 8px; border: 1px solid #dddddd; text-align: left;'>min_samples_leaf</th>")
            report_file.write("<th style='padding: 8px; border: 1px solid #dddddd; text-align: left;'>n_neighbors</th>")
            
            report_file.write("<th style='padding: 8px; border: 1px solid #dddddd; text-align: left;'>Score</th>")
            report_file.write("<th style='padding: 8px; border: 1px solid #dddddd; text-align: left;'>Accuracy</th>")
            report_file.write("<th style='padding: 8px; border: 1px solid #dddddd; text-align: left;'>Precision</th>")
            report_file.write("<th style='padding: 8px; border: 1px solid #dddddd; text-align: left;'>Recall</th>")
            report_file.write("<th style='padding: 8px; border: 1px solid #dddddd; text-align: left;'>F1</th>")
            # report_file.write("<th style='padding: 8px; border: 1px solid #dddddd; text-align: left;'>Start Time</th>")
            # report_file.write("<th style='padding: 8px; border: 1px solid #dddddd; text-align: left;'>End Time</th>")
            report_file.write("<th style='padding: 8px; border: 1px solid #dddddd; text-align: left;'>Duration</th>")
            
            
            # Add more table headers as needed
            report_file.write("</tr>")

            # Iterate through the rows and format the values, date, and time
            for _, row in select_results.iterrows():

                model = row['params_classifier']
                criterion = row['params_criterion']
                n_estimators = row['params_n_estimators']
                max_depth = row['params_max_depth']
                min_samples_split = row['params_min_samples_split']
                min_samples_leaf = row['params_min_samples_leaf']
                n_neighbors = row['params_n_neighbors']

                formatted_value = f"{row['value']:.2f}"
                formatted_accuracy = f"{row['accuracy']:.2f}"
                formatted_precision = f"{row['precision']:.2f}"
                formatted_recall = f"{row['recall']:.2f}"
                formatted_f1 = f"{row['f1']:.2f}"

                # start_time = row['datetime_start'].strftime("%Y-%m-%d %H:%M:%S")
                # end_time = row['datetime_complete'].strftime("%Y-%m-%d %H:%M:%S")
                
                duration = row['duration']
                
                report_file.write("<tr>")
                report_file.write(f"<td style='padding: 8px; border: 1px solid #dddddd;'>{model}</td>")
                report_file.write(f"<td style='padding: 8px; border: 1px solid #dddddd;'>{criterion}</td>")
                report_file.write(f"<td style='padding: 8px; border: 1px solid #dddddd;'>{n_estimators}</td>")
                report_file.write(f"<td style='padding: 8px; border: 1px solid #dddddd;'>{max_depth}</td>")
                report_file.write(f"<td style='padding: 8px; border: 1px solid #dddddd;'>{min_samples_split}</td>")
                report_file.write(f"<td style='padding: 8px; border: 1px solid #dddddd;'>{min_samples_leaf}</td>")
                report_file.write(f"<td style='padding: 8px; border: 1px solid #dddddd;'>{n_neighbors}</td>")
                report_file.write(f"<td style='padding: 8px; border: 1px solid #dddddd;'>{formatted_value}</td>")
                report_file.write(f"<td style='padding: 8px; border: 1px solid #dddddd;'>{formatted_accuracy}</td>")
                report_file.write(f"<td style='padding: 8px; border: 1px solid #dddddd;'>{formatted_precision}</td>")
                report_file.write(f"<td style='padding: 8px; border: 1px solid #dddddd;'>{formatted_recall}</td>")
                report_file.write(f"<td style='padding: 8px; border: 1px solid #dddddd;'>{formatted_f1}</td>")
                # report_file.write(f"<td style='padding: 8px; border: 1px solid #dddddd;'>{start_time}</td>")
                # report_file.write(f"<td style='padding: 8px; border: 1px solid #dddddd;'>{end_time}</td>")
                report_file.write(f"<td style='padding: 8px; border: 1px solid #dddddd;'>{duration}</td>")
                
                # Add more table data cells as needed
                report_file.write("</tr>")

            report_file.write("</table>")


            # Hyperparamete importances###############################################################
            try:
                fig = optuna.visualization.matplotlib.plot_param_importances(study)

                # Salvar a imagem em um buffer de memória
                image_buffer = io.BytesIO()
                fig.get_figure().savefig(image_buffer, format="png")
                image_buffer.seek(0)
                # Salva a imagem no sistema de arquivos
                image_path = "param_importances.png"
                with open(image_path, "wb") as f:
                    f.write(image_buffer.getvalue())
                # Converter a imagem no buffer em base64
                param_importances = base64.b64encode(image_buffer.read()).decode()

                # Escreva a imagem no relatório HTML
                report_file.write("<h1>Hyperparameter Importances</h1>")
                report_file.write("<div style='display:flex; justify-content:center;'>")
                report_file.write(f'<img $safeprojectname$="data:image/png;base64,{param_importances}" style="margin: 10 auto;">')
                report_file.write("</div>")



                # Hyperparamete optimization_history #################################################################
                fig2 = optuna.visualization.matplotlib.plot_optimization_history(study)

                # Salvar a imagem em um buffer de memória
                image_buffer = io.BytesIO()
                fig2.get_figure().savefig(image_buffer, format="png")
                image_buffer.seek(0)
                # Salva a imagem no sistema de arquivos
                image_path = "optimization_history.png"
                with open(image_path, "wb") as f:
                    f.write(image_buffer.getvalue())
                # Converter a imagem no buffer em base64
                optimization_history = base64.b64encode(image_buffer.read()).decode()

                # Escreva a imagem no relatório HTML
                report_file.write("<h1>Hyperparameter Optimization History</h1>")
                report_file.write("<div style='display:flex; justify-content:center;'>")
                report_file.write(f'<img $safeprojectname$="data:image/png;base64,{optimization_history}" style="margin: 10 auto;">')
                report_file.write("</div>")
       
            except Exception as e:
                self.logger.error(f"Erro ao gerar gráficos de visualização: {e}")
                # Caso ocorra um erro, apenas ignore a geração dos gráficos
                pass

            ######################################################################################################
            # Top Models Section #################################################################################
            report_file.write("<h1>Best Model</h1>")

            # Create a table with CSS styles
            report_file.write("<table style='border-collapse: collapse; width: 50%;'>")
            report_file.write("<tr style='background-color: #f2f2f2;'>")
            report_file.write("<th style='padding: 8px; border: 1px solid #dddddd; text-align: left;'>Model Name</th>")
            report_file.write("<th style='padding: 8px; border: 1px solid #dddddd; text-align: left;'>Best Parameters</th>")
            # Add more table headers as needed
            report_file.write("</tr>")

            # Insert the values into the table
            report_file.write("<tr>")
            report_file.write(f"<td style='padding: 8px; border: 1px solid #dddddd;'>{model_name}</td>")
            report_file.write(f"<td style='padding: 8px; border: 1px solid #dddddd;'>{model_params}</td>")
            # Add more table data cells as needed
            report_file.write("</tr>")

            report_file.write("</table>")


            # Report Metrics ######################################################################################
            report_file.write("<h1>Classification Report</h1>") 
            report_df = pd.DataFrame(report).transpose()

            # Aplicar estilo CSS para alinhar os valores à direita
            html_table = report_df.to_html(classes='styled-table', justify='center')
            html_table = html_table.replace('<th>Malware</th>', '<th style="text-align:right;">Malware</th>')
            html_table = html_table.replace('<th>Benign</th>', '<th style="text-align:right;">Benign</th>')
            # Aplicar estilo CSS para formatar as outras colunas
            report_df['precision'] = report_df['precision'].apply(lambda x: f'{x:.2f}')
            report_df['recall'] = report_df['recall'].apply(lambda x: f'{x:.2f}')
            report_df['f1-score'] = report_df['f1-score'].apply(lambda x: f'{x:.2f}')
            report_df['support'] = report_df['support'].apply(lambda x: f'{x:.2f}')

            # Converter o DataFrame em HTML
            html_table = report_df.to_html()

            # Escrever a tabela no relatório
            report_file.write(html_table)


            # Confusion Matrix Section ############################################################################
         
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative Class", "Positive Class"])
            fig, ax = plt.subplots()
            disp.plot(cmap=plt.cm.Blues, values_format="d", ax=ax)
            ax.set_title("Confusion Matrix")

            # Renderizar o gráfico em memória e converter para base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            base64_image = base64.b64encode(buffer.read()).decode()
            plt.close()

            # Adicionar a imagem da matriz de confusão ao relatório HTML
            report_file.write("<h1>Confusion Matrix</h1>")
            report_file.write("<div style='display:flex; justify-content:center;'>")
            report_file.write(f'<img $safeprojectname$="data:image/png;base64,{base64_image}" style="margin: 0 auto;">')
            report_file.write("</div>")

            # ROC Curve Section ######################################################################################
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)

            # Plotar a curva ROC 
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc='lower right')

            # Renderizar o gráfico da curva ROC em memória e converter para base64
            roc_curve_buffer = io.BytesIO()
            plt.savefig(roc_curve_buffer, format="png")
            roc_curve_buffer.seek(0)
            base64_roc_curve = base64.b64encode(roc_curve_buffer.read()).decode()
            plt.close()

            # Adicionar o gráfico da curva ROC com a informação AUC centralizado ao relatório HTML
            report_file.write("<h1>ROC Curve</h1>")
            report_file.write("<p>A curva ROC (Receiver Operating Characteristic) é uma ferramenta essencial na avaliação do desempenho de um modelo de detecção de malwares Android. Ela representa a taxa de verdadeiros positivos (TPR) em função da taxa de falsos positivos (FPR) para diferentes valores de threshold. No contexto de detecção de malwares, a curva ROC nos ajuda a entender como o modelo está realizando a classificação entre arquivos maliciosos e benignos, permitindo-nos avaliar seu poder discriminativo e escolher o ponto de operação ideal com base nas necessidades específicas do cenário de segurança.</p>")
            report_file.write("<div style='display:flex; justify-content:center;'>")
            report_file.write(f'<p style="text-align:center;">AUC: {roc_auc:.2f}</p>')
            report_file.write(f'<img $safeprojectname$="data:image/png;base64,{base64_roc_curve}" style="margin: 0 auto;">')
            report_file.write("</div>")



            # Precision-Recall Curve Section ####################################################################
            # Gerar a curva Precision-Recall
            precision, recall, _ = precision_recall_curve(y_test, y_pred)

            # Plotar a curva Precision-Recall
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2)
            plt.fill_between(recall, precision, color='blue', alpha=0.2)  # Adicionar área sombreada sob a curva
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')

            plt.grid(True)  # Adicionar grade para melhorar a visualização dos valores
            plt.tight_layout()  # Ajustar layout para evitar sobreposição de rótulos

            precision_recall_curve_buffer = io.BytesIO()
            plt.savefig(precision_recall_curve_buffer, format="png")
            precision_recall_curve_buffer.seek(0)
            base64_precision_recall_curve = base64.b64encode(precision_recall_curve_buffer.read()).decode()
            plt.close()

            # Adicionar o gráfico da curva Precision-Recall centralizado ao relatório HTML
            report_file.write("<h1 style='text-align: center;'>Precision-Recall Curve</h1>")
            report_file.write("<p>A curva Precision-Recall é outra métrica crucial na avaliação de modelos de detecção de malwares Android, especialmente em casos onde há desbalanceamento entre as classes. Ela mostra a precisão (Precision) em função do recall para diferentes valores de threshold. Em detecção de malwares, onde a precisão na identificação de arquivos maliciosos é essencial, a curva Precision-Recall nos permite avaliar a capacidade do modelo de evitar falsos positivos, fornecendo uma visão mais detalhada do desempenho do modelo em diferentes níveis de recall. Isso é particularmente importante em cenários de segurança, onde a identificação correta de arquivos maliciosos é prioritária, mesmo que isso signifique um aumento nos falsos positivos.</p>")
            report_file.write("<div style='display:flex; justify-content:center;'>")
            report_file.write(f'<img $safeprojectname$="data:image/png;base64,{base64_precision_recall_curve}">')
            report_file.write("</div>")


            report_file.write("<h1 style='text-align: center;'>Interpretability</h1>")


            # Gráfico de importância de features gerado pelo LIME

            report_file.write("<div>")
            report_file.write("<p>O Gráfico de importância de features gerado pelo LIME fornece uma representação visual das características mais importantes para o modelo preditivo. Cada barra no gráfico representa uma característica do conjunto de dados, e a altura da barra indica a importância relativa dessa característica para a predição do modelo.</p>")
            report_file.write(f"<img $safeprojectname$='{lime_exp_filepath}' width='100%' height='600'></img>")
            report_file.write("</div>")
            report_file.write("<br>")


            # Gráfico de interpretação gerado pelo LIME
            report_file.write("<div>")
            report_file.write("<p>Esses valores fornecem uma visão detalhada de como cada feature influencia a predição do modelo em relação à classificação de um arquivo como malware ou benigno.</p>")

            report_file.write(f"<iframe $safeprojectname$='{exp_filepath}' width='100%' height='600'></iframe>")
            report_file.write("</div>")
            report_file.write("<br>")

           

            # Gráfico de interpretação gerado pelo SHAP
            report_file.write("<div>")
            report_file.write("<p>Force Plot (gráfico de força): Este é um gráfico individual que mostra as contribuições de cada feature para uma determinada previsão. É útil para explicar previsões individuais e entender quais features estão impulsionando a classificação de um malware específico como benigno ou malicioso.</p>")
            report_file.write(f"<iframe $safeprojectname$='{shap_exp_filepath}' width='100%' height='600'></iframe>")
            report_file.write("</div>")
            report_file.write("<br>")

            # Write the HTML footer
            report_file.write("</body></html>")

        print(f"Unified report generated successfully at '{report_filename}'")

        self.save_performance('Evaluation', start_time_step6, start_memory_step6)
        self.export_to_csv('performance_summary.csv')
        self.plot_performance('performance_metrics.jpg')
                ###########################################################################
        dataset: PandasDataset = mlflow.data.from_pandas(dataset_df, source=self.dataset_url)
        # Registrando os melhores hiperparâmetros no MLflow
        with mlflow.start_run(run_name=run_name,log_system_metrics=True) as run:
            mlflow.set_tag("experiment_type", "Auto_ML")           
            mlflow.log_input(dataset, context="training")
            mlflow.log_params(model_params)
            # Registre as métricas do relatório de classificação
            mlflow.log_metric("accuracy", report["accuracy"])
            mlflow.log_metric("precision", report["macro avg"]["precision"])
            mlflow.log_metric("recall", report["macro avg"]["recall"])
            mlflow.log_metric("f1", report["macro avg"]["f1-score"])
            mlflow.log_metric("mcc", matthews_corrcoef(y_test, y_pred))
            #mlflow.log_metric(key="train_loss", value=train_loss, step=epoch, timestamp=now)


            #mlflow.log_artifact(self.dataset_url, artifact_path="datasets")

            # Log de artefatos em cada diretório
            
            self.log_artifact_if_exists(estimator_html_repr(pipeline), "00_Data_info")
            self.log_artifact_if_exists("results/missing_values_heatmap.png", "01_preprocessing")
            self.log_artifact_if_exists("results/clean_missing_values_heatmap.png", "01_preprocessing")
            self.log_artifact_if_exists("results/pca_biplot.png", "02_feature_engineering")
            self.log_artifact_if_exists("results/lasso_feature_importance.png", "02_feature_engineering")
            self.log_artifact_if_exists("results/train_test_distribution.png", "02_feature_engineering")
            self.log_artifact_if_exists("results/Hyperparameters_Results.csv", "03_model_optimization")
            self.log_artifact_if_exists("results/Models_Ranking.csv", "03_model_optimization")
            mlflow.log_artifact(image_path, "03_model_optimization")
            self.log_artifact_if_exists('performance_metrics.jpg', "04_evaluation_metrics")
            self.log_artifact_if_exists(model_filepath, "04_evaluation_metrics")
            self.log_artifact_if_exists(model_filepath, "04_evaluation_metrics")
            self.log_artifact_if_exists("shap_summary_plot.png", "05_interpretability")
            self.log_artifact_if_exists("shap_beeswarm_plot.png", "05_interpretability")
            self.log_artifact_if_exists("lime_feature_importance.jpg", "05_interpretability")
            self.log_artifact_if_exists("interpretability_0.html", "05_interpretability")

            # Verificação adicional para os arquivos específicos
            if lime_exp_filepath and os.path.exists(lime_exp_filepath):
                mlflow.log_artifact(lime_exp_filepath, artifact_path="05_interpretability")
            if exp_filepath and os.path.exists(exp_filepath):
                mlflow.log_artifact(exp_filepath, artifact_path="05_interpretability")

            # Log do relatório se existir
            if report_filename and os.path.exists(report_filename):
                mlflow.log_artifact(report_filename, artifact_path="report")

            # Log da tabela de resultados se o dataframe não for None
            if report_df is not None:
                mlflow.log_table(data=report_df, artifact_file="results.csv")
            # Log the model
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="MH_Best_Model",
                registered_model_name=model_name,
            )

        run = mlflow.get_run(mlflow.last_active_run().info.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        dataset_source.load()
        mlflow.end_run() 

        print("acess : http://localhost:5000")

        
        self.logger.info("Done!")
