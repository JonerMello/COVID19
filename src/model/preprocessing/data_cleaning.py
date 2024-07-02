import logging
from colorama import Fore, Style
import pandas as pd
import numpy as np
import re
import os
import plotly.io as pio
import plotly.graph_objs as go
import plotly.offline as pyo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import matthews_corrcoef
from model.preprocessing.data_info import DataInfo

class DataCleaning(BaseEstimator, TransformerMixin):
    """
    A data cleaning transformer for preprocessing datasets.
    """

    def __init__(self, remove_duplicates=False, remove_missing_values=False, remove_outliers=False, label=None):
        """
        Initialize the DataCleaning transformer.

        Args:
            remove_duplicates (bool): Whether to remove duplicate rows.
            remove_missing_values (bool): Whether to remove rows with missing values.
            remove_outliers (bool): Whether to perform outlier removal.
            label (str): The label for the dataset.
        """
        self.remove_duplicates = remove_duplicates
        self.remove_missing_values = remove_missing_values
        self.remove_outliers = remove_outliers
        self.label = label
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def fit(self, dataset, label=None):
        """
        Fit method for the transformer. No actual fitting is performed.

        Args:
            dataset (pd.DataFrame): The input dataset.
            label (str, optional): The label for the dataset.

        Returns:
            self
        """
        return self

    def transform(self, dataset, label=None):
        """
        Transform the input dataset based on the specified cleaning steps.

        Args:
            dataset (pd.DataFrame): The input dataset.
            label (str, optional): The label for the dataset.

        Returns:
            pd.DataFrame: The cleaned dataset.
        """
        
        if self.remove_outliers:
            dataset = self.remove_outliers_step(dataset)
        if self.remove_duplicates:
            dataset = self.remove_duplicates_step(dataset)
        if self.remove_missing_values:
            dataset = self.remove_missing_values_step(dataset)
        return dataset

    def remove_outliers_step(self, dataset):
        """
        Remove outliers from the dataset.

        Args:
            dataset (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The dataset with outliers removed.
        """
        try:
            logging.info("Remove Outliers...")
            dataset = dataset.applymap(self.custom_convert)
            return dataset
        except Exception as e:
            colored_message = f"[{Fore.RED}Error while removing outliers: {e}{Style.RESET_ALL}]"
            self.logger.error(colored_message)
            return None

    def remove_duplicates_step(self, dataset):
        """
        Remove duplicate rows from the dataset.

        Args:
            dataset (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The dataset with duplicate rows removed.
        """
        try:
            data_info = DataInfo(self.label, dataset)
            logging.info("Remove Duplicates...")

            crypto_col = data_info.find_and_drop_crypto_column()

            if crypto_col:
                dataset.drop_duplicates(subset=[crypto_col], inplace=True)
                dataset.drop(columns=[crypto_col], inplace=True)
            else:
                colored_message = f"[{Fore.YELLOW}No columns with cryptographic signatures found.{Style.RESET_ALL}]"
                self.logger.warning(colored_message)
            return dataset
        except Exception as e:
            colored_message = f"[{Fore.RED}Error while removing duplicates: {e}{Style.RESET_ALL}]"
            self.logger.error(colored_message)
            return None

    def remove_missing_values_step(self, dataset):
        """
        Remove rows with missing values from the dataset.

        Args:
            dataset (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The dataset with missing values removed.
        """
        self.plot_missing_values_heatmap(dataset)
        try:
            logging.info("Remove Missing Values...")
            dataset.replace('?', np.nan, inplace=True)
            dataset.dropna(axis=1, how='all', inplace=True)
            dataset.dropna(how='any', inplace=True)
            dataset = dataset.loc[:, (dataset != 0).any(axis=0)]
            self.plot_missing_values_heatmap(dataset)
            return dataset
            
        except Exception as e:
            colored_message = f"[{Fore.RED}Error while removing missing values: {e}{Style.RESET_ALL}]"
            self.logger.error(colored_message)
            return None

    def custom_convert(self, value):
        """
        Custom conversion function for handling data types.

        Args:
            value: The input value.

        Returns:
            int, float, or np.nan: The converted value.
        """
        
        if isinstance(value, int) or isinstance(value, bool):
            return int(value)
        elif isinstance(value, str):
            lower_value = value.lower()
            if lower_value == 'true':
                return 1
            elif lower_value == 'false':
                return 0
            elif lower_value == '?':
                return np.nan
        return value

    def plot_distribution_before_after_outliers_removal(self, dataset, results_folder="results"):
        """
        Plot distribution of numeric columns before and after outlier removal.

        Args:
            dataset (pd.DataFrame): The input dataset.
            results_folder (str): The folder to save the HTML file.

        Returns:
            None
        """
        numeric_cols = dataset.select_dtypes(include=np.number).columns
    
        # Calcula a distribuição antes da remoção de outliers
        before_outliers_removal = dataset[numeric_cols].copy()
        before_outliers_removal.describe()
    
        # Remove outliers
        after_outliers_removal = self.remove_outliers_step(dataset)
    
        # Calcula a distribuição após a remoção de outliers
        after_outliers_removal = after_outliers_removal[numeric_cols].copy()
        after_outliers_removal.describe()
    
        # Plota os gráficos de distribuição
        plt.figure(figsize=(12, 6))
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(2, len(numeric_cols), i)
            sns.histplot(before_outliers_removal[col], kde=True, color='blue', bins=30)
            plt.title(f'Distribution Before Outliers Removal - {col}')
            plt.xlabel('')
            plt.ylabel('Frequency')
            plt.subplot(2, len(numeric_cols), i+len(numeric_cols))
            sns.histplot(after_outliers_removal[col], kde=True, color='orange', bins=30)
            plt.title(f'Distribution After Outliers Removal - {col}')
            plt.xlabel('')
            plt.ylabel('Frequency')
    
        plt.tight_layout()
    
        # Criar o caminho completo para o arquivo HTML
        html_file_path = os.path.join(results_folder, "outliers_removal_distribution.png")
    
        # Salvar o gráfico como um arquivo HTML
        plt.savefig(html_file_path, format='png')
    
        # Fechar a figura para liberar memória
        plt.close()

        return html_file_path



    def plot_missing_values_heatmap(self, dataset, results_folder="results"):
        """
        Plot a heatmap of missing values in the dataset.

        Args:
            dataset (pd.DataFrame): The input dataset.
            results_folder (str): The folder to save the heatmap.

        Returns:
            None
        """
        # Calcula a matriz de booleanos indicando valores ausentes
        if dataset.isnull().values.any():

            # Calcula a matriz de booleanos indicando valores ausentes
            missing_values = dataset.isnull()

            # Plota o heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(missing_values, cmap='viridis', cbar=False)
            plt.title('Missing Values Heatmap')
            plt.xlabel('Columns')
            plt.ylabel('Rows')
            # Cria o caminho completo para o arquivo de imagem
            heatmap_file_path = os.path.join(results_folder, "missing_values_heatmap.png")

            # Salva o gráfico como um arquivo PNG
            plt.savefig(heatmap_file_path, bbox_inches='tight')

            # Fecha a figura para liberar memória
            plt.close()
        else:
           # Criar uma matriz de booleanos com todas as entradas sendo False
            clean_values = dataset.notnull()

            # Plota o heatmap com cores uniformes
            plt.figure(figsize=(10, 6))
            sns.heatmap(clean_values, cmap='viridis', cbar=False)
            plt.title('Clean Dataset Heatmap')
            plt.xlabel('Columns')
            plt.ylabel('Rows')
            # Cria o caminho completo para o arquivo de imagem
            heatmap_file_path = os.path.join(results_folder, "clean_missing_values_heatmap.png")

            # Salva o gráfico como um arquivo PNG
            plt.savefig(heatmap_file_path, bbox_inches='tight')

            # Fecha a figura para liberar memória
            plt.close()

        return heatmap_file_path