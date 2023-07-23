import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    LearningCurveDisplay,
    ValidationCurveDisplay,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from joblib import dump, load

tor_traffic_undersampled = pd.read_csv(
    "../../data/processed/undersampled/tor-traffic-proc-undersampled.csv"
)

tor_traffic_undersampled["label"] = tor_traffic_undersampled["label"].astype("category")

# Encoding Protocol.
oh_enc = OneHotEncoder()

protocol_transf = oh_enc.fit_transform(
    tor_traffic_undersampled["Protocol"].to_numpy().reshape(-1, 1)
)

encoded_protocol = pd.DataFrame(
    protocol_transf.toarray(),
    columns=list(["protocol_"] + oh_enc.get_feature_names_out()),
)
tor_traffic_undersampled = pd.concat(
    [encoded_protocol, tor_traffic_undersampled], axis=1
)

# Removing features to address suspected data leakge.
X = tor_traffic_undersampled[
    tor_traffic_undersampled.columns.difference(
        ["Source IP", "Source Port", "Destination IP", "Destination Port", "label"],
        sort=False,
    )
]
y = tor_traffic_undersampled["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7212023, stratify=y
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.25, random_state=7212023, stratify=y_train
)

# ――――――――――――――――――――――
# LOGISTIC REGRESSION
# ――――――――――――――――――――――
lr = LogisticRegression(random_state=7212023, penalty="l2")
scores_lr_train = cross_val_score(lr, X_train, y_train, cv=5, scoring="f1")
np.mean(scores_lr_train)

lr.fit(X_train, y_train)
preds_lr_valid = lr.predict(X_valid)
print(classification_report(preds_lr_valid, y_valid))

# Plotting Learning Curves
LearningCurveDisplay.from_estimator(lr, X_train, y_train)
plt.title("Logistic Regression Learning Curve - lr-mdl-1")
plt.savefig("../models/learning_curves/lr-lc1.png")

# ――――――――――――――
# RANDOM FOREST
# ――――――――――――――
# Starting with a base case for Random forest. It looks to perform better than lr-mdl-1.
rf_base = RandomForestClassifier(
    n_estimators=300, max_depth=5, min_samples_leaf=3, random_state=7212023
)
scores_rf_base = cross_val_score(rf_base, X_train, y_train, cv=5, scoring="f1")
scores_rf_base_cv_acc = np.mean(scores_rf_base)
scores_rf_base_cv_std = np.std(scores_rf_base)

print(f"Accuracy: {scores_rf_base_cv_acc} +/- {scores_rf_base_cv_std}")

rf_base.fit(X_train, y_train)
preds_rf_valid = rf_base.predict(X_valid)

print(classification_report(preds_rf_valid, y_valid))

LearningCurveDisplay.from_estimator(rf_base, X_train, y_train)
plt.title("Random Forest Learning Curve - rf-mdl-1")
plt.savefig("../models/learning_curves/rf-lc1.png")

# ――――――――――――――――――――――
# HYPERPARAMETER TUNING
# ――――――――――――――――――――――
max_depth_range = np.arange(2, 15, 2)
min_samples_split_range = np.arange(2, 15, 2)
min_samples_leaf_range = np.arange(2, 10, 2)
n_est_range = [100, 300, 500]

param_grid = {
    "n_estimators": n_est_range,
    "max_depth": max_depth_range,
    "min_samples_split": min_samples_split_range,
    "min_samples_leaf": min_samples_leaf_range,
}

grid_rf = RandomForestClassifier()
gs = GridSearchCV(
    estimator=grid_rf,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    refit=True,
    n_jobs=-1,
)

gs = gs.fit(X_train, y_train)

print(gs.best_score_)
print(gs.best_params_)

rf = gs.best_estimator_

LearningCurveDisplay.from_estimator(rf, X_train, y_train)
plt.title("Random Forest Learning Curve - rf-mdl-2")
plt.savefig("../models/learning_curves/rf-lc2.png")

# Looking at Validation Curve to improve max_depth parameter.
ValidationCurveDisplay.from_estimator(
    rf, X_train, y_train, param_name="n_estimators", param_range=[100, 300, 500]
)
plt.title("Random Forest Validation Curve - rf-mdl-2")
plt.figtext(
    0.5,
    0.001,
    "(param: max_depth, range = [2, 4, 6, 8, 10, 12])",
    horizontalalignment="center",
)
plt.savefig("../models/validation_curves/rf-vc2.png")

rf_best_params = {
    "max_depth": 14,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "min_samples_split": 2,
    "n_estimators": 300,
}

rf.set_params(**rf_best_params)
rf.fit(X_train, y_train)
rf.score(X_valid, y_valid)

# ―――――――――――――――――――――――――――――――――
# ASSESSING FEATURE IMPORTANCES
# ―――――――――――――――――――――――――――――――――
rf_feature_importances = pd.DataFrame(
    {"Feature": rf.feature_names_in_, "Importance": rf.feature_importances_}
)
rf_feature_importances.sort_values(by="Importance", ascending=True, inplace=True)

rf_feature_importances.set_index("Feature").plot(kind="barh", color="#67bdfb")
plt.title("Random Forest Feature Importances - rf-mdl-2")
plt.savefig("../models/rf-mdl-2-feature-importances.png")

# ―――――――――――――――――――――――――――――――――――――――――――――――
# CALCULATING TEST ACCURACY AND CONFUSION MATRIX
# ―――――――――――――――――――――――――――――――――――――――――――――――
preds_test = rf.predict(X_test)

print(classification_report(preds_test, y_test))

test_acc = round(accuracy_score(y_test, preds_test), 4)

# Achieved a test accuracy of 98.04%
ConfusionMatrixDisplay(confusion_matrix(y_test, preds_test)).plot(cmap="Blues")
plt.title(f"Confusion Matrix - rf-mdl-2, Test Acc: {test_acc}")
plt.savefig("../models/confusion-matrix-rf-mdl-2.png")

dump(rf, "../../models/rf-mdl-2.model")
