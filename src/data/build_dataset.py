# ――――――――――――――――――――――――――――――――――――――――――
# PREPROCESSING TOR NETWORK TRAFFIC DATA 2023-07-21
# ――――――――――――――――――――――――――――――――――――――――――
# Dataset Source: https://www.unb.ca/cic/datasets/tor.html
# Scenario-A -> SelectedFeatures-10s-TOR-NonTor
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split

traffic_data = pd.read_csv(
    "../../data/raw/tor-traffic-data/Scenario-A/SelectedFeatures-10s-TOR-NonTOR.csv"
)
traffic_data.head()

traffic_data.info()

# Removing the space at the start of each column name.
traffic_data.columns = [c.lstrip() for c in traffic_data.columns]

traffic_data_proc = traffic_data.copy()

# ――――――――――――――――――――――――――――
# IDENTIFYING MISSING VALUES
# ――――――――――――――――――――――――――――
# Flow Bytes has 2 missing values - idx 67832, 67833 and there are also some inf values.
traffic_data_proc[traffic_data_proc.isnull().any(axis=1)]
traffic_data_proc.replace(np.inf, np.nan, inplace=True)
traffic_data_proc.dropna(how="any", inplace=True)

traffic_data_proc.isnull().sum()

traffic_data_proc["label"] = traffic_data_proc["label"].map({"nonTOR": 0, "TOR": 1})

# Only ~12% of the network traffic is Tor traffic.
traffic_data_proc["label"].value_counts(normalize=True) * 100

# ―――――――――――――
# PORT NUMBERS
# ―――――――――――――
traffic_data_proc["Source Port"].value_counts()
traffic_data_proc["Destination Port"].value_counts()

# ―――――――――――――――――
# PROTOCOL NUMBERS
# ―――――――――――――――――
# There only looks to be 2 values for the Protocol number (6, 17).
# TCP (Transmission Control Protocol): Provides reliable and ordered communication between devices.
# Tor uses TCP as its transport protocol -> https://wiki.wireshark.org/Tor.md
# We also see this in the data as all instances of Tor traffic is using the TCP protocol.
traffic_data_proc["Protocol"].value_counts(normalize=True) * 100

# Note: Source IP, Destination IP, Source Port, and Destination Port will not be considered
# when building the model due to potential data leakage.

traffic_data_proc.to_csv("../../data/processed/tor-traffic-proc.csv", index=False)

X = traffic_data_proc[traffic_data_proc.columns.difference(["label"], sort=False)]
y = traffic_data_proc["label"]

# ―――――――――――――――――――――――――
# HANDLING CLASS IMBALANCE
# ―――――――――――――――――――――――――
# Due to the class imbalance of Tor network traffic being the minority class, we will use undersampling
# to handle the class imbalance.

rus = RandomUnderSampler(random_state=7212023)
X_train_resampled_under, y_train_resampled_under = rus.fit_resample(X, y)

traffic_data_proc_undersampled = pd.concat(
    [X_train_resampled_under, y_train_resampled_under], axis=1
)

traffic_data_proc_undersampled.to_csv(
    "../../data/processed/undersampled/tor-traffic-proc-undersampled.csv", index=False
)

# TODO:
# smote = SMOTE(random_state=7212023)
# X_train_resampled_over, y_train_resampled_over = smote.fit_resample(X_train, y_train)

# traffic_data_proc_oversampled = pd.concat(
#     [X_train_resampled_over, y_train_resampled_over], axis=1
# )

# traffic_data_proc_oversampled.to_csv(
#     "../../data/processed/oversampled/tor-traffic-proc-oversampled.csv", index=False
# )
