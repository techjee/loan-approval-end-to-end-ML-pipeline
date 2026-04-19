import pandas as pd
import numpy as np
import os


class DataPreprocessing:
    def __init__(self, file_path):
        self.file_path = file_path

    def preprocess(self):
        df = pd.read_csv(self.file_path)

        print("Before preprocessing:\n")
        print(df.isnull().sum())

        # Drop Loan_ID (not useful)
        df.drop(columns=["Loan_ID"], inplace=True)

        # Fill missing values

        df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
        df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
        df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
        df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])

        df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
        df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())
        df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

        print("\nAfter handling missing values:\n")
        print(df.isnull().sum())

        # Feature Engineering

        df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
        df["IncomeLoanRatio"] = df["TotalIncome"] / df["LoanAmount"]

        # Encode target

        df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

        return df


if __name__ == "__main__":
    file_path = os.path.join("data", "raw", "loan_data.csv")

    processor = DataPreprocessing(file_path)
    df = processor.preprocess()

    print("\nProcessed Data:\n")
    print(df.head())