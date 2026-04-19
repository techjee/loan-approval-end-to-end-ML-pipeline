import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

plt.close("all")
sns.set(style="whitegrid")


class ResultVisualizer:
    def __init__(self, file_path):
        self.file_path = file_path
        os.makedirs("artifacts/plots", exist_ok=True)

    def load_and_prepare_data(self):
        df = pd.read_csv(self.file_path)

        # 1. Missing values before preprocessing
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

        if not missing_counts.empty:
            plt.figure(figsize=(8, 5))
            missing_counts.plot(kind="bar")
            plt.title("Missing Values Before Preprocessing")
            plt.xlabel("Features")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("artifacts/plots/missing_values_before.png")
            plt.show()

        # 2. Loan approval distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(x="Loan_Status", data=df)
        plt.title("Loan Approval Distribution")
        plt.xlabel("Loan Status")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("artifacts/plots/loan_status_distribution.png")
        plt.show()

        # Drop Loan_ID
        df = df.drop(columns=["Loan_ID"])

        # Handle missing values
        df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
        df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
        df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
        df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])

        df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
        df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())
        df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

        # Feature engineering
        df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
        df["IncomeLoanRatio"] = df["TotalIncome"] / df["LoanAmount"]

        # Encode target
        df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

        X = df.drop(columns=["Loan_Status"])
        y = df["Loan_Status"]

        return X, y

    def run_visualization(self):
        X, y = self.load_and_prepare_data()

        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }

        results = []
        best_pipeline = None
        best_name = None
        best_f1 = 0
        best_y_pred = None

        # 3. ROC curve comparison
        plt.figure(figsize=(8, 6))

        for name, model in models.items():
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "ROC-AUC": roc_auc
            })

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

            if f1 > best_f1:
                best_f1 = f1
                best_name = name
                best_pipeline = pipeline
                best_y_pred = y_pred

        plt.plot([0, 1], [0, 1], "r--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig("artifacts/plots/roc_curve_comparison.png")
        plt.show()

        # 4. Model performance comparison
        results_df = pd.DataFrame(results)
        results_melted = results_df.melt(
            id_vars="Model", var_name="Metric", value_name="Score"
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(data=results_melted, x="Metric", y="Score", hue="Model")
        plt.title("Model Performance Comparison")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig("artifacts/plots/model_comparison.png")
        plt.show()

        print("\nModel Comparison Table:\n")
        print(results_df)
        print(f"\nBest Model: {best_name} with F1 Score: {best_f1:.4f}")

        # 5. Best model confusion matrix only
        cm = confusion_matrix(y_test, best_y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {best_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("artifacts/plots/best_model_confusion_matrix.png")
        plt.show()

        # 6. Feature importance / coefficient importance
        model = best_pipeline.named_steps["model"]
        fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
        feature_names = fitted_preprocessor.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(15)

        else:
            coefs = model.coef_[0]
            feature_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": abs(coefs)
            }).sort_values(by="Importance", ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_df, x="Importance", y="Feature")
        plt.title(f"Top 15 Influential Features - {best_name}")
        plt.tight_layout()
        plt.savefig("artifacts/plots/feature_importance.png")
        plt.show()

        print("\n✅ All final best visuals generated successfully!")


if __name__ == "__main__":
    file_path = os.path.join("data", "raw", "loan_data.csv")
    visualizer = ResultVisualizer(file_path)
    visualizer.run_visualization()