import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preprocessing
def load_data(path):
    """Load CSV dataset into pandas DataFrame."""
    return pd.read_csv(path, sep=';')

def encode_categorical(df):
    """Encode categorical features using LabelEncoder."""
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

def preprocess_data(df, target="dropout"):
    """Split features and target, scale numerical features."""
    X = df.drop(columns=[target, 'G3'])

    # old results that included G3:
    # X = df.drop(columns=[target])

    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, X.columns

# Model Training
def train_models(X_train, y_train):
    """Train multiple models and return them."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

# Evaluation
def evaluate_models(models, X_test, y_test):
    """Evaluate trained models and print performance metrics."""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1]) if len(np.unique(y_test)) == 2 else None
        results[name] = {"accuracy": acc, "auc": auc}

        print(f"===== {name} =====")
        print("Accuracy:", acc)
        if auc:
            print("ROC AUC:", auc)
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return results

# Visualization
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10,6))
        sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
        plt.title("Feature Importance")
        plt.show()

# Main Pipeline
def main():
    # Load data (example: Math dataset)
    df = load_data("student-mat.csv")
    print("Data Loaded:", df.shape)

    # Encode categorical
    df, encoders = encode_categorical(df)

    # Binary classification target: dropout risk (e.g., G3 < 10 -> 1 else 0)
    df['dropout'] = (df['G3'] < 10).astype(int)

    X, y, scaler, feature_names = preprocess_data(df, target='dropout')

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    models = train_models(X_train, y_train)

    # Evaluate
    results = evaluate_models(models, X_test, y_test)

    # Visualize feature importance (Random Forest example)
    plot_feature_importance(models["Random Forest"], feature_names)

if __name__ == "__main__":

    main()
