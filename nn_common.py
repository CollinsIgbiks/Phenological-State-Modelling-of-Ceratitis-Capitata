"""
nn_common.py
------------
Shared wrapper for the two neural networks so RNN and LSTM are trained in
exactly the same way; the ONLY difference between them is the recurrent layer.

- inputs standardised per feature (fit on training data only)
- targets standardised (log1p counts -> z-scores)
- last 15% of the training period (chronologically) used for early stopping
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras

SEED = 42
UNITS = 32
DROPOUT = 0.2
EPOCHS = 300
BATCH = 16
PATIENCE = 25
LR = 1e-3


class RecurrentForecaster:
    def __init__(self, cell="lstm"):
        self.cell = cell

    def _build(self, lookback, n_feat, horizon):
        keras.utils.set_random_seed(SEED)
        Layer = keras.layers.LSTM if self.cell == "lstm" else keras.layers.SimpleRNN
        model = keras.Sequential([
            keras.Input(shape=(lookback, n_feat)),
            Layer(UNITS),
            keras.layers.Dropout(DROPOUT),
            keras.layers.Dense(horizon),
        ])
        model.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")
        return model

    def fit(self, X, Y):
        n, L, f = X.shape
        self.x_mu = X.reshape(-1, f).mean(0); self.x_sd = X.reshape(-1, f).std(0) + 1e-8
        self.y_mu = Y.mean(); self.y_sd = Y.std() + 1e-8
        Xs = (X - self.x_mu) / self.x_sd
        Ys = (Y - self.y_mu) / self.y_sd

        n_val = max(int(0.15 * n), 5)            # chronological validation split (no shuffling)
        self.model = self._build(L, f, Y.shape[1])
        self.model.fit(
            Xs[:-n_val], Ys[:-n_val],
            validation_data=(Xs[-n_val:], Ys[-n_val:]),
            epochs=EPOCHS, batch_size=BATCH, verbose=0, shuffle=True,
            callbacks=[keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)],
        )
        return self

    def predict(self, X):
        Xs = (X - self.x_mu) / self.x_sd
        return self.model.predict(Xs, verbose=0) * self.y_sd + self.y_mu
