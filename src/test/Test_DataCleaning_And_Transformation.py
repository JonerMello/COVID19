import unittest
import pandas as pd
import numpy as np
from model.preprocessing.data_transformation import DataTransformation
from model.preprocessing.data_cleaning import DataCleaning
from sklearn.pipeline import Pipeline
import numpy as np
class TestDataCleaningAndTransformation(unittest.TestCase):
    def test_transform_pipeline(self):
        # Criando um dataset de teste



        data = {
            'Feature1': [1, 1, None, 1, True, 0, 1, None, 1, True, 0, 1, None, 1, True, 0, 1, None, 1, True],
            'Feature2': [0, 1, np.nan, 0, False, 1, 0, np.nan, 1, False, 0, 1, np.nan, 0, False, 1, 0, np.nan, 1, False],
            'Feature3': [np.nan, 1, 1, 0, '?', 1, 0, 1, 0, '?', np.nan, 1, 1, 0, '?', 1, 0, 1, 0, '?'],
            'Feature4': [0, 1, 1, 0, '?', 1, 0, 1, 0, '?', 0, 1, 1, 0, '?', 1, 0, 1, 0, '?'],
            'Feature5': [5, 1, 10, 1, '?', 5, 1, 10, 1, '?', 5, 1, 10, 1, '?', 5, 1, 10, 1, '?'],
            'sha256': [
                '8994a47e0315feb75c1a9e1bd7487caf24d5a7677c5e05cdb218208f9bd3922b',
                '4d8f5a95d189a978c8a29c88cee32dbd87468a6cb839c32057931dccfbd0b6dd',
                'c5924075a0d17d45602913ee1d9fb02fa6166777434fafc0152b672f2e1c1575',
                '46070d4bf934fb0d4b06d9e2c46e346944e322444900a435d7d9a95e6d7435f5',
                '5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5',
                '5cc6b70f94e08cc2e2b02d0f2b2b16ad44dce190875d5c2b259c02c26c566ac7',
                '7c740048993b5086e173c69fde9ce3ee1b6523ad066e5a9cfe9b4653ca7a5fbf',
                'a1f23fdef50848ef0f84e96a31e3db2e4b6a01659993f5f3b15a8c1a1e8de181',
                'e67e22a2e00e6e5e9c6155b3c585f4aa123045f2f8ea739eed15aecd9934a831',
                '4e73f1be0c6f89f3a3b7ed2484a0b3f24a4a0e36e6cf491e86006c40e0c56b1c',
                '8994a47e0315feb75c1a9e1bd7487caf24d5a7677c5e05cdb218208f9bd3922b',
                '4d8f5a95d189a978c8a29c88cee32dbd87468a6cb839c32057931dccfbd0b6dd',
                'c5924075a0d17d45602913ee1d9fb02fa6166777434fafc0152b672f2e1c1575',
                '46070d4bf934fb0d4b06d9e2c46e346944e322444900a435d7d9a95e6d7435f5',
                '5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5',
                '5cc6b70f94e08cc2e2b02d0f2b2b16ad44dce190875d5c2b259c02c26c566ac7',
                '7c740048993b5086e173c69fde9ce3ee1b6523ad066e5a9cfe9b4653ca7a5fbf',
                'a1f23fdef50848ef0f84e96a31e3db2e4b6a01659993f5f3b15a8c1a1e8de181',
                'e67e22a2e00e6e5e9c6155b3c585f4aa123045f2f8ea739eed15aecd9934a831',
                '4e73f1be0c6f89f3a3b7ed2484a0b3f24a4a0e36e6cf491e86006c40e0c56b1c'
            ],
            'Label': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive', 'Positive', 'Negative', 'Positive', 'Negative', 'Positive',
              'Positive', 'Negative', 'Positive', 'Negative', 'Positive', 'Positive', 'Negative', 'Positive', 'Negative', 'Positive']
            }


        df = pd.DataFrame(data)
        # Gerar dados duplicados
        dataset = pd.concat([df] * 2, ignore_index=True)

        print(dataset)
        print("______________________________________")
        # Criando uma instância do pipeline com DataCleaning
        preprocessor = Pipeline(steps=[
            ('Data Cleaning', DataCleaning(remove_duplicates=True, remove_missing_values=True, remove_outliers=True)),
            ('Data Transformation', DataTransformation(label="Label", one_hot_encoder=False,do_label_encode=True))
        ])

        # Chamando o método transform do pipeline
        cleaned_X, cleaned_y = preprocessor.fit_transform(dataset)
        
        # Verificando os resultados para X
        expected_X = pd.DataFrame( {
            'Feature1': [1, 1, 0, 1, 1],
            'Feature2': [1, 0, 1, 0, 1],
            'Feature3': [1, 0, 1, 0, 0],
            'Feature4': [1, 0, 1, 0, 0],
            'Feature5': [1, 1, 5, 1, 1]
        }).values  # Convertendo para array NumPy
        print("Expected X\n",cleaned_X.astype(np.int8))
        self.assertTrue(np.array_equal(cleaned_X, expected_X))

        # Verificando os resultados para y
        expected_y_encoded = np.array([0, 0, 1, 0, 0])  # Label encoded form of 'Negative'
        print("Expected y\n",cleaned_y.astype(np.int8))
        self.assertTrue(np.array_equal(cleaned_y.astype(np.int8), expected_y_encoded))

if __name__ == '__main__':
    unittest.main()




