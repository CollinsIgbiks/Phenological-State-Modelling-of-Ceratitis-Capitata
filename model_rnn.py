"""
model_rnn.py
------------
Simple (Elman) recurrent neural network: one SimpleRNN layer reading the last
14 days, followed by a Dense layer that outputs the next 7 days at once.

Run:  python model_rnn.py
"""
from evaluate import run_model
from nn_common import RecurrentForecaster


def RNNForecaster():
    return RecurrentForecaster(cell="rnn")


if __name__ == "__main__":
    run_model("RNN", RNNForecaster)
