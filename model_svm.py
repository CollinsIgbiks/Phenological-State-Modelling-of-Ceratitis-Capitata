"""
model_svm.py
------------
Support Vector Regression. SVR has no notion of sequence, so the 14-day
window is flattened into one long feature vector. One SVR per forecast day
(MultiOutputRegressor). Features are standardised on the training data only.

Run:  python model_svm.py
"""
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from evaluate import run_model

# Hyper-parameters: tune C / epsilon / gamma on the training period if you have time
C, EPSILON, GAMMA = 3.0, 0.1, "scale"


class SVMForecaster:
    def __init__(self):
        self.model = MultiOutputRegressor(
            make_pipeline(StandardScaler(), SVR(kernel="rbf", C=C, epsilon=EPSILON, gamma=GAMMA))
        )

    def fit(self, X, Y):
        self.model.fit(X.reshape(len(X), -1), Y)
        return self

    def predict(self, X):
        return self.model.predict(X.reshape(len(X), -1))


if __name__ == "__main__":
    run_model("SVM", SVMForecaster)
