"""
model_lstm.py
-------------
Long Short-Term Memory network: identical to model_rnn.py except the recurrent
layer is an LSTM, whose gates let it keep information (e.g. accumulated heat)
over longer spans than a plain RNN.

Run:  python model_lstm.py
"""
from evaluate import run_model
from nn_common import RecurrentForecaster


def LSTMForecaster():
    return RecurrentForecaster(cell="lstm")


if __name__ == "__main__":
    run_model("LSTM", LSTMForecaster)
