import argparse
import logging
from controller.core import Core
from logo import *

class Main:
    """
    Main class for running the core functionality.

    Args:
        dataset_url (str): URL or file path to the dataset.
        label (str): Name of the target column.
        log_level (int, optional): Logging level (default: logging.INFO).
        remove_duplicates (bool, optional): Whether to remove duplicate rows (default: True).
        remove_missing_values (bool, optional): Whether to remove rows with missing values (default: True).
        remove_outliers (bool, optional): Whether to remove outliers (default: True).
        one_hot_encoder (bool, optional): Whether to use one-hot encoding (default: True).
        do_label_encode (bool, optional): Whether to perform label encoding (default: True).

    Attributes:
        dataset_url (str): URL or file path to the dataset.
        label (str): Name of the target column.
        log_level (int): Logging level.
    """

    def __init__(self, dataset_url, label, log_level=logging.INFO, remove_duplicates=True,
                 remove_missing_values=True, remove_outliers=True, one_hot_encoder=True, do_label_encode=True,balance_classes=True):
        self.dataset_url = dataset_url
        self.label = label
        self.log_level = log_level
        self.remove_duplicates = remove_duplicates
        self.remove_missing_values = remove_missing_values
        self.remove_outliers = remove_outliers
        self.one_hot_encoder = one_hot_encoder
        self.do_label_encode = do_label_encode
        self.balance_classes = balance_classes

    def run_core(self):
        """
        Initialize and run the Core module with the specified arguments.
        """
        core = Core(self.dataset_url, self.label, log_level=self.log_level,
                    remove_duplicates=self.remove_duplicates, remove_missing_values=self.remove_missing_values,
                    remove_outliers=self.remove_outliers, one_hot_encoder=self.one_hot_encoder,
                    do_label_encode=self.do_label_encode,balance_classes=self.balance_classes)
        core.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example of using argparse in the Main class.')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='URL or file path to the dataset.')
    parser.add_argument('--label-column', '-l', type=str, required=True, help='Name of the target column.')
    parser.add_argument('--log-level', type=str, default='info', choices=['debug', 'info'], help='Logging level.')
    parser.add_argument('--remove-duplicates', action='store_false', default=True, help='Remove duplicate rows.')
    parser.add_argument('--remove-missing-values', action='store_false', default=True, help='Remove rows with missing values.')
    parser.add_argument('--remove-outliers', action='store_false', default=True, help='Remove outliers.')
    parser.add_argument('--one-hot-encoder', action='store_false', default=True, help='Apply one-hot encoding.')
    parser.add_argument('--label-encode', action='store_false', default=True, help='Apply label encoding.')
    parser.add_argument('--balance-classes', action='store_false', default=True, help='Apply balance classes.')

   

    print(logo)
    args = parser.parse_args()

    main = Main(args.dataset, args.label_column, log_level=args.log_level,
                remove_duplicates=args.remove_duplicates, remove_missing_values=args.remove_missing_values,
                remove_outliers=args.remove_outliers, one_hot_encoder=args.one_hot_encoder,
                do_label_encode=args.label_encode,balance_classes=args.balance_classes)
    main.run_core()




    #main = Main("https://raw.githubusercontent.com/Malware-Hunter/sbseg22-feature-selection/main/datasets/drebin_215_all.csv", "class")
    #main = Main("androcrawl.csv", "Detection Ratio")