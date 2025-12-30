import os
import sys
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class MachineLearningModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def load_data(self):
        try:
            data = pd.read_csv(self.data_path)
            return data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            sys.exit(1)

    def preprocess_data(self, data):
        X = data.drop('target', axis=1)
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        try:
            self.model.fit(X_train, y_train)
        except Exception as e:
            logging.error(f"Error training model: {e}")
            sys.exit(1)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)
        return accuracy, report, matrix

def main():
    data_path = 'data.csv'
    model = MachineLearningModel(data_path)
    data = model.load_data()
    X_train, X_test, y_train, y_test = model.preprocess_data(data)
    model.train_model(X_train, y_train)
    accuracy, report, matrix = model.evaluate_model(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.3f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()