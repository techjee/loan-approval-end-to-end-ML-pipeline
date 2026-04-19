import pandas as pd
import os


class DataIngestion:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        try:
            df = pd.read_csv(self.file_path)
            print("Dataset loaded successfully!\n")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None


if __name__ == "__main__":
    file_path = os.path.join("data", "raw", "loan_data.csv")

    ingestion = DataIngestion(file_path)
    df = ingestion.load_data()

    if df is not None:
        print("First 5 rows:\n")
        print(df.head())

        print("\nDataset Info:\n")
        print(df.info())

        print("\nSummary Statistics:\n")
        print(df.describe())