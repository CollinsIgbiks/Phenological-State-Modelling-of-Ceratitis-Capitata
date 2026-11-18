# Phenological State Modelling of *Ceratitis capitata*

Predicting the emergence of new generations of the Mediterranean fruit fly (*Ceratitis capitata*) in citrus, using climate data and machine learning.

## Overview

Knowing when a new pest generation will emerge lets growers time their control measures. This project models the phenological state of *C. capitata*, a major pest of citrus trees, and examines how climate and host-plant phenology affect its development. The project investigates whether new-generation activity can be forecast…” new-generation activity within a **7-day window** from time-series data of hourly temperature and daily trap catches.

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

## Results

All models were tested with walk-forward validation on September–December 2020 (one trap, one season), forecasting new trap catches for the next 7 days.

| Model | MAE, next-7-day total (flies) | Correlation with actual |
|---|---|---|
| 7-day moving average (baseline) | 9.1 | 0.73 |
| Persistence (baseline) | 14.8 | 0.66 |
| SVM | 16.0 | 0.21 |
| LSTM | 17.3 | 0.02 |
| RNN | 17.7 | 0.07 |

**None of the three models outperformed the simple baselines, and none predicted the November generation peak.** The models learned only from the August peak, which occurred in hot weather. The November rise happened as temperatures fell, and was likely driven by citrus ripening rather than heat accumulation. With only 165 days of data and two generation peaks, the neural networks in particular had too few examples to generalise. The effect of plant phenology could not be tested because the dataset contains no plant data.

### Limitations and future work
- A single season from a single trap; more seasons or locations are the main need
- No host-plant phenology data
- Thresholds of 15/35 °C were used; published lower thresholds for *C. capitata* are around 10–13 °C
- Next steps: frame the task as yes/no generation-onset prediction, tune hyperparameters, and add longer-term degree-day features