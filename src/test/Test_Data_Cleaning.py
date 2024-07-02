import unittest
import pandas as pd
from model.preprocessing.data_cleaning import DataCleaning
from model.preprocessing.data_info import DataInfo
from sklearn.pipeline import Pipeline
import numpy as np
class TestDataCleaning(unittest.TestCase):
    def test_transform_pipeline(self):
        # Criando um dataset de teste
        data = {
            'Feature1': [1 , 0 , None , 1, True],
            'Feature2': [0 , 1 , np.nan , 0, False],
            'Feature3': [ np.nan , 1 , 1 , 0, '?'],
            'Feature4': [ 0 , 1 , 1 , 0, '?'],
            'Feature5': [ 5 , 1 , 10 , 1, '?'],
            'Label': ['Positive', 'Negative', 'Positive', 'Negative','Positive']
            
        }
        df = pd.DataFrame(data)
        # Gerar dados duplicados
        dataset = pd.concat([df] * 2, ignore_index=True)

        print(dataset)
        print("______________________________________")
        # Criando uma instância do pipeline com DataCleaning
        preprocessor = Pipeline(steps=[
            ('Data Cleaning', DataCleaning(remove_outliers=True, remove_duplicates=True, remove_missing_values=True))
        ])
        # Chamando o método transform do pipeline
        cleaned_dataset = preprocessor.fit_transform(dataset)
        print(cleaned_dataset)
       

        # Verificando os resultados
        expected_dataset = pd.DataFrame({
            'Feature1': [0, 1, 0, 1],
            'Feature2': [1, 0, 1, 0],
            'Feature3': [1, 0, 1, 0],
            'Feature4': [1, 0, 1, 0],
            'Feature5': [1, 1, 1, 1],
            'Label': ['Negative', 'Negative','Negative', 'Negative']
        }).values  # Convertendo para array NumPy

        self.assertTrue(np.array_equal(cleaned_dataset.values, expected_dataset))

if __name__ == '__main__':
    unittest.main()