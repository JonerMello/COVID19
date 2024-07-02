import os
import pandas as pd
import logging
import datetime
import mlflow
from colorama import Fore, Style

class DatasetValidation:
    def __init__(self, dataset_url, label):
        self.dataset_url = dataset_url
        self.label = label
        self.logger = logging.getLogger(__name__)  # Inicialize o logger
    
    def validate_dataset(self):
        # Verificar se o conjunto de dados é válido
        if not os.path.exists(self.dataset_url):
            self.logger.error(f"Dataset directory '{self.dataset_url}' does not exist.")
            return False
        
        # Verificar se o arquivo CSV existe
        if not os.path.exists(self.dataset_url):
            self.logger.error(f"CSV file '{self.dataset_url}' does not exist.")
            return False
        
        # Verificar se o conjunto de dados é válido
        dataset_df = pd.read_csv(self.dataset_url, encoding="utf8")
        if dataset_df is None or dataset_df.empty:
            self.logger.error("Invalid dataset. Please provide a valid dataset CSV file.")
            return False
        
        # Verificar se a coluna do rótulo existe no conjunto de dados
        if self.label not in dataset_df.columns:
            self.logger.error(f"Label column '{self.label}' not found in the dataset.")
            return False
        
        # Verificar se o arquivo possui a extensão .csv
        if not self.dataset_url.lower().endswith('.csv'):
            self.logger.error("Invalid file format. Expected a .csv file.")
            return False
        
        return True
   
    def load_data(self):
        try:
            dataset =  pd.read_csv(self.dataset_url, encoding="utf8", low_memory=True)
            return dataset
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            return None