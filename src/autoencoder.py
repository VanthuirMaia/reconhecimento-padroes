import numpy as np
from tensorflow.keras import models, layers

def create_autoencoder(input_dim=30, bottleneck_dim=3):
    # Entrada
    input_layer = layers.Input(shape=(input_dim,))

    # Codificador
    encoded = layers.Dense(16, activation="relu")(input_layer)
    encoded = layers.Dense(8, activation="relu")(encoded)
    bottleneck = layers.Dense(bottleneck_dim, activation="relu")(encoded)

    # Decodificador
    decoded = layers.Dense(8, activation="relu")(bottleneck)
    decoded = layers.Dense(16, activation="relu")(decoded)
    output_layer = layers.Dense(input_dim, activation="linear")(decoded)

    # Modelos
    autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
    encoder = models.Model(inputs=input_layer, outputs=bottleneck)

    # Compilar
    autoencoder.compile(optimizer="adam", loss="mse")

    return autoencoder, encoder


def train_autoencoder(autoencoder, X_scaled, epochs=40, batch_size=16):
    history = autoencoder.fit(
        X_scaled, X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history

