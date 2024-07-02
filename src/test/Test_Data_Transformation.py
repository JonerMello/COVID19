import unittest
import logging
import pandas as pd
import numpy as np
from model.preprocessing.data_transformation import DataTransformation
from sklearn.pipeline import Pipeline

class TestDataTransformation(unittest.TestCase):
    def test_transform_pipeline(self):
       # Criando um dataset de teste
        data = {
            'Feature1': [1, 0, 1, 1],
            'Feature2': [False, False, True, True],
            'Feature3': [True, False, True, False],
            'Label': ['Mal', 'Ben', 'Ben', 'Mal']
        }
        dataset = pd.DataFrame(data)
        print(dataset)
        print("________________________________")
        # Criando uma instância do pipeline com DataTransformation
        preprocessor = Pipeline(steps=[
            ('Data Transformation', DataTransformation(label="Label", one_hot_encoder=True, do_label_encode=True))
        ])

        # Chamando o método fit_transform do pipeline
        transformation = preprocessor.fit_transform(dataset)
        X, y = transformation[0], transformation[1]
    
        # Verificando os resultados de X
        expected_X = pd.DataFrame({
            'Feature1': [1, 0, 1, 1],
            'Feature2': [0, 0, 1, 1],
            'Feature3': [1, 0, 1, 0],
     
        }).values  # Convertendo para array NumPy
       
        self.assertTrue(np.array_equal(X, expected_X))
        #print(expected_y)
        expected_y = np.array([1, 0, 0, 1])
        self.assertTrue(np.array_equal(y, expected_y))
        dataset_new = pd.concat([pd.DataFrame(expected_X), pd.DataFrame(expected_y)], axis=1)
        print(dataset_new)
if __name__ == '__main__':
    unittest.main()
