"""
Retreina o autoencoder e salva em formato nativo Keras (.keras).
TF deve ser importado antes do sklearn para evitar conflito de DLL no Windows.
"""
import os, sys, pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# TF primeiro — conflito de DLL com sklearn se importado depois
import tensorflow as tf
from src.autoencoder import create_autoencoder, train_autoencoder

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- dados (mesmo split do notebook 04) ---
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Treino: {X_train_scaled.shape}  Teste: {X_test_scaled.shape}")

# --- arquitetura idêntica ao notebook 04 ---
autoencoder, encoder = create_autoencoder(input_dim=30, bottleneck_dim=3)
autoencoder.summary()

history = train_autoencoder(autoencoder, X_train_scaled, epochs=40)

loss_final = round(history.history["loss"][-1], 4)
print(f"\nTreino concluído. Loss final: {loss_final}")

# --- salvar em formato nativo (.keras) ---
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(models_dir, exist_ok=True)

encoder_path = os.path.join(models_dir, "encoder.keras")
history_path = os.path.join(models_dir, "ae_history.pkl")

encoder.save(encoder_path)
print(f"Encoder salvo: {encoder_path}")

with open(history_path, "wb") as f:
    pickle.dump(history.history, f)
print(f"Histórico salvo: {history_path}")

# --- sanidade ---
enc_loaded = tf.keras.models.load_model(encoder_path)
out = enc_loaded.predict(X_test_scaled[:3], verbose=0)
assert out.shape == (3, 3), f"Shape inesperado: {out.shape}"
print(f"Sanidade OK — predict shape: {out.shape}")
print("\n=== encoder.keras pronto ===")
