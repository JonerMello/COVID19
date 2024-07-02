import logging
import numpy as np
import pandas as pd
import re
from tabulate import tabulate
from colorama import Fore, Style
from sklearn.metrics import matthews_corrcoef
import platform
import psutil
import re

class DataInfo:
    def __init__(self, label, dataset):
        """
        Initialize DataInfo object.

        Args:
            label (str): The label or target variable name.
            dataset (pd.DataFrame): The dataset to analyze.
        """
        self.label = label
        self.dataset = dataset
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO) 
        # Initialize attributes to store results
        self.system_info_result = None
        self.info_table_result = None
        self.data_types_result = None
        self.balance_info_result = None
        self.duplicates_missing_result = None
        self.features_info_result = None

    def display_dataframe_info(self):
        """
        Display various information about the dataset.
        """
        self.system_info_result = self.system_info()
        self.info_table_result = self.display_info_table()
        self.data_types_result = self.display_data_types()
        self.balance_info_result = self.display_balance_info()
        self.duplicates_missing_result = self.display_duplicates_missing()
        self.features_info_result = self.display_features_info()

    def system_info(self):
        try:
            # Get the operating system version
            system_version = platform.platform()

            # Get information about RAM memory
            memory_info = psutil.virtual_memory()

            # Create a table to display system information
            system_info_table = pd.DataFrame({
                "Operating System Version": [system_version],
                "Total RAM Memory Usage (GB)": [memory_info.total / (1024 ** 3)],
                "Available RAM Memory (GB)": [memory_info.available / (1024 ** 3)],
                "Used RAM Memory (GB)": [memory_info.used / (1024 ** 3)]
            })

            # Display the system information table
            system_info_table_str = tabulate(system_info_table, headers="keys", tablefmt="psql", showindex=False)
            self.logger.info("System Information:\n%s\n", system_info_table_str)
            return system_info_table
        except Exception as e:
            colored_message = f"[{Fore.RED}Error while displaying system info: {e}{Style.RESET_ALL}]"
            self.logger.error(colored_message)
            return None


    def display_info_table(self):
        """
        Display the basic information about the dataset (number of rows and columns).
        """
        try:
            info_table = pd.DataFrame({
                "Rows": [self.dataset.shape[0]],
                "Columns": [self.dataset.shape[1]]
            })
            info_table_str = tabulate(info_table, headers="keys", tablefmt="psql", showindex=False)
            self.logger.info("DataFrame Size:\n%s\n", info_table_str)
            return info_table
        except Exception as e:
            colored_message = f"[{Fore.RED}Error while displaying info table: {e}{Style.RESET_ALL}]"
            self.logger.error(colored_message)
            return None

    def display_data_types(self):
        """
        Display the data types of columns in the dataset.
        """
        try:
            types_counts = self.dataset.dtypes.value_counts()
            types_table = pd.DataFrame({"Data Type": types_counts.index, "Count": types_counts.values})
            types_table["Data Type"] = types_table["Data Type"].apply(lambda x: x.name)
            types_table_str = tabulate(types_table, headers="keys", tablefmt="psql")
            self.logger.info("Data types:\n%s\n", types_table_str)
            return types_table
        except Exception as e:
            colored_message = f"[{Fore.RED}Error while displaying data types: {e}{Style.RESET_ALL}]"
            self.logger.error(colored_message)
            return None

    def display_balance_info(self):
        """
        Display information about class balance (for classification tasks).
        """
        try:
            balance = self.dataset[self.label].value_counts(normalize=True) * 100
            balance.index.name = "Label" 
            balance.name = "Percentage"
            balance_table = balance.reset_index()
            # Format the 'Percentage' column as a percentage with two decimal places and the '%' symbol
            balance_table["Percentage"] = balance_table["Percentage"].apply(lambda x: f"{x:.2f}%")
            balance_table_str = tabulate(balance_table, headers="keys", tablefmt="psql", showindex=False)
            self.logger.info("Balancing:\n%s\n", balance_table_str)
            return balance_table
        except Exception as e:
            colored_message = f"[{Fore.RED}Error while displaying balance info: {e}{Style.RESET_ALL}]"
            self.logger.error(colored_message)
            return None


    def display_duplicates_missing(self):
        """
        Display information about duplicate values and missing values in the dataset.
        """
        try:
            found_col = self.find_and_drop_crypto_column()
            duplicate_count = self.calculate_duplicate_count(found_col)

            info_table = pd.DataFrame({
                "Number of duplicate data": [duplicate_count],
                "Number of null values": [self.dataset.isnull().any().sum()]
            })
            info_table_str = tabulate(info_table, headers="keys", tablefmt="psql", showindex=False)
            self.logger.info("Duplicates and Missing Values:\n%s\n", info_table_str)
            return info_table
        except Exception as e:
            colored_message = f"[{Fore.RED}Error while displaying duplicates and missing values: {e}{Style.RESET_ALL}]"
            self.logger.error(colored_message)
            return None

    def display_features_info(self):
        """
        Display information about features in the dataset, such as permissions and API calls.
        """
        try:
            permissions_found = self.has_android_permissions()
            apicalls_found = self.has_android_api_calls()

            info_table = pd.DataFrame({
                "Permissions found": [len(permissions_found)],
                "API_Calls found": [len(apicalls_found)]
            })
            info_table_str = tabulate(info_table, headers="keys", tablefmt="psql", showindex=False)
            self.logger.info("Features Information:\n%s\n", info_table_str)

            found_col = self.find_and_drop_crypto_column()
            if found_col:
                colored_message = f"[{Fore.YELLOW}Column '{found_col}' has cryptographic signature.{Style.RESET_ALL}]"
                self.logger.warning(colored_message)
            return info_table
        except Exception as e:
            colored_message = f"[{Fore.RED}Error while displaying features info: {e}{Style.RESET_ALL}]"
            self.logger.error(colored_message)
            return None

    def calculate_duplicate_count(self, found_col):
        """
        Calculate the number of duplicate rows in a specific column.

        Args:
            found_col (str): The name of the column to check for duplicates.

        Returns:
            int: The number of duplicate rows in the column.
        """
        duplicate_count = 0
        if found_col is not None:
            duplicate_count = self.dataset[found_col].duplicated().sum()
        return duplicate_count

    def is_balanced_dataset(self):
        """
        Check if a dataset is balanced for classification tasks.

        Returns:
            bool: True if the dataset is balanced, False otherwise.
        """
        label_counts = {}
        for label in self.label:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        label_count_values = list(label_counts.values())
        majority_class_count = max(label_count_values)
        minority_class_count = min(label_count_values)
        balanced_ratio = minority_class_count / majority_class_count
        if balanced_ratio >= 0.5:
            return True
        else:
            return False

    def has_categorical_rows(self):
        """
        Check if the dataframe contains rows with categorical values.

        Returns:
            bool: True if there are rows with categorical values, False otherwise.
        """
        categorical_cols = self.dataset.select_dtypes(include=['category', 'object']).columns
        has_categorical = self.dataset[categorical_cols].apply(lambda x: x.str.contains('[A-Za-z]', regex=True)).any(axis=1).any()
        return has_categorical

    def has_android_permissions(self):
        """
        Check if the dataset contains Android permissions.

        Returns:
            list: List of Android permissions found in the dataset.
        """
        android_permission_regex = r'(.*([Pp][Ee][Rr][Mm][Ii][Ss][Ss][Ii][Oo][Nn].*)|\b[A-Za-z_\d]*[A-Z]\b)'
        permissions_found = []
        for data in self.dataset:
            match = re.search(android_permission_regex, data)
            if match:
                permission = match.group(0)
                permissions_found.append(permission)
        return permissions_found

    def has_android_api_calls(self):
        """
        Check if the dataset contains Android API calls.

        Returns:
            list: List of Android API calls found in the dataset.
        """
        android_apicalls_regex = r'.*(Landroid|Ljava).*' 
        apicalls_found = []
        for data in self.dataset:
            match = re.search(android_apicalls_regex, data)
            if match:
                apicalls = match.group(0)
                apicalls_found.append(apicalls)
        return apicalls_found

    def is_crypto_signature(self, data):
        """
        Check if a string represents a cryptographic signature.

        Args:
            data (str): The string to check.

        Returns:
            bool: True if the string is a cryptographic signature, False otherwise.
        """
        sha256_pattern = r'^[a-fA-F0-9]{64}$'  # SHA-256 pattern
        md5_pattern = r'^[a-fA-F0-9]{32}$'     # MD5 pattern
        if re.match(sha256_pattern, data) or re.match(md5_pattern, data):
            return True
        else:
            return False

    def find_and_drop_crypto_column(self):
        """
        Find and drop a column if it contains a cryptographic signature.

        Returns:
            str: The name of the column containing the cryptographic signature, or None if not found.
        """
        found_column = None
        for index, row in self.dataset.iterrows():
            for column in self.dataset.columns:
                cell_value = str(row[column])
                if self.is_crypto_signature(cell_value):
                    found_column = column
                    break
            if found_column:
                break  # Stop searching after finding the first occurrence
        return found_column


    def save_data_info_to_html(self, output_file_path):
        """
        Save the information generated by DataInfo to an HTML file.

        Args:
            output_file_path (str): The path to the HTML file where the information will be saved.
        """
        try:
           

            system_info = self.system_info_result.to_html(index=False)
            info_table = self.info_table_result.to_html(index=False)

            data_types = self.data_types_result.to_html(index=False)
            balance_info = self.balance_info_result.to_html(index=False)
            duplicates_missing = self.duplicates_missing_result.to_html(index=False)
            features_info = self.features_info_result.to_html(index=False)

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

            # Save the content to an HTML file
            with open("system_info.html", "w") as html_file:
                html_file.write(html_content)


            return html_content  
        except Exception as e:
            self.logger.error(f"Error while saving data info to HTML: {e}")
            return ""
       

    