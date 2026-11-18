# Phenological State Modelling of *Ceratitis capitata*

Predicting the emergence of new generations of the Mediterranean fruit fly (*Ceratitis capitata*) in citrus, using climate data and machine learning.

## Overview

Knowing when a new pest generation will emerge lets growers time their control measures. This project models the phenological state of *C. capitata*, a major pest of citrus trees, and examines how climate and host-plant phenology affect its development. The goal is to forecast new-generation activity within a **7-day window** from time-series data of hourly temperature and daily trap catches.

Three algorithms are implemented and compared:

- **SVM**: Support Vector Machine (regression)
- **RNN**: Recurrent Neural Network
- **LSTM**: Long Short-Term Memory network

The models are evaluated with walk-forward validation and compared against simple baseline forecasts.

## Repository structure

| File | Purpose |
|---|---|
| `data_prep.py` | Cleans the raw data, aggregates it to daily values, computes degree-days and builds the model inputs |
| `evaluate.py` | Walk-forward evaluation, error metrics and plots shared by all models |
| `model_svm.py` | SVM model |
| `model_rnn.py` | RNN model |
| `model_lstm.py` | LSTM model |
| `nn_common.py` | Shared training setup for the RNN and LSTM |
| `compare.py` | Runs all models and baselines and produces the comparison |

## Getting started

```bash
pip install -r requirements.txt
python compare.py          # run every model and the comparison
python model_lstm.py       # or run a single model
```

Results (plots and metrics) are saved to `results/`.