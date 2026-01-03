import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    scaler = StandardScaler()
    data[['feature1', 'feature2']] = scaler.fit_transform(data[['feature1', 'feature2']])
    return data

def train_model(data):
    X = data[['feature1', 'feature2']]
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred, y_test

def evaluate_model(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def main():
    parser = argparse.ArgumentParser(description='Machine Learning Model')
    parser.add_argument('-f', '--file', help='Path to the data file', required=True)
    args = parser.parse_args()
    data = load_data(args.file)
    data = preprocess_data(data)
    model, y_pred, y_test = train_model(data)
    accuracy, report = evaluate_model(y_pred, y_test)
    print(f"Accuracy: {accuracy}")
    print(report)

if __name__ == "__main__":
    main()