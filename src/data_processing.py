from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Carrega o dataset e divide em treino e teste.
def load_data(test_size=0.2, random_state=42):
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# Normaliza dadosnúmericos (necessário para PCA).
def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Retorna os nomes das features do dataset.
def get_feature_names():
    data = load_breast_cancer()
    return data.feature_names