import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

from sklearn.base import BaseEstimator, TransformerMixin

class DataTransformation(BaseEstimator, TransformerMixin):
    """
    A custom data transformation class for preprocessing data.

    Args:
        label (str): The name of the target label column.
        one_hot_encoder (bool): Whether to perform one-hot encoding.
        do_label_encode (bool): Whether to perform label encoding.

    Attributes:
        label (str): The name of the target label column.
        one_hot_encoder (bool): Whether to perform one-hot encoding.
        do_label_encode (bool): Whether to perform label encoding.
        logger (Logger): Logger object for logging information.

    """

    def __init__(self, label=None, one_hot_encoder=False, do_label_encode=False):
        self.label = label
        self.one_hot_encoder = one_hot_encoder
        self.do_label_encode = do_label_encode
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def fit(self, dataset, label=None):
        """
        Fit method for the transformer. It doesn't perform any actual training.

        Args:
            dataset (DataFrame): The input dataset.
            label (str, optional): The name of the target label column.

        Returns:
            self: Returns the transformer object itself.

        """
        return self

    def transform(self, dataset, label=None):
        """
        Transform method to preprocess the dataset.

        Args:
            dataset (DataFrame): The input dataset.
            label (str, optional): The name of the target label column.

        Returns:
            X (DataFrame): The preprocessed feature matrix.
            y (Series): The target labels.

        """
        y = dataset[self.label]
        X = dataset.drop([self.label], axis=1)
        
        if self.one_hot_encoder:
            """
            one_hot_encoder: Encode categorical features into a single numeric matrix.
            Takes a string 'label' representing a classification column in the dataset and a dataset.
            Returns the processed values of X and y.
            """

            logging.info("OneHotEncoder...")

            X = self.one_hot_encode(X)

        if self.do_label_encode:
            logging.info("LabelEncode...")
            y = self.label_encode(y)

        

        return X, y

    def one_hot_encode(self, X):
        """
        Perform one-hot encoding on categorical features in the input DataFrame.

        Args:
            X (DataFrame): The input DataFrame containing categorical features.

        Returns:
            X_encoded_df (DataFrame): The DataFrame with one-hot encoded categorical features.

        """
        try:
            # Separate numeric and boolean columns
            X_numeric_bool = X.select_dtypes(include=[np.number, np.bool_])

            # Remove as colunas em que todos os valores são maiores que 1
            X_numeric_bool = X_numeric_bool.loc[:, (X_numeric_bool <= 1).all()]
            # Apply OneHotEncoder to numeric and boolean columns
            onehot_encoder = OneHotEncoder(
                categories='auto',
                drop='first',
                sparse=False,
                handle_unknown='ignore'
            )
            X_encoded = onehot_encoder.fit_transform(X_numeric_bool)

            # Get column names after encoding
            encoded_column_names = onehot_encoder.get_feature_names_out(input_features=X_numeric_bool.columns)

            # Convert the encoded matrix back to a DataFrame
            X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_column_names)
            return X_encoded_df
        except Exception as e:
            colored_message = f"[Error in one-hot encoding: {e}]"
            self.logger.error(colored_message)
            return None

    def label_encode(self, y):
        """
        Perform label encoding on the target labels.

        Args:
            y (Series): The target labels.

        Returns:
            y_encoded (array): The encoded target labels.

        """
        try:
            y_encoded = LabelEncoder().fit_transform(y)
            return y_encoded
        except Exception as e:
            colored_message = f"[Error in label encoding: {e}]"
            self.logger.error(colored_message)
            return None

   