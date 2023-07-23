import os

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("ggplot")
figures_path = os.path.join(os.path.dirname(__file__), "figures")

# ―――――――――――――――――――――――――――――
# VISUALIZING PROCESSED DATA
# ―――――――――――――――――――――――――――――
traffic_data_proc = pd.read_csv("../../data/processed/tor-traffic-proc.csv")

# Visualizing the target variable
sns.histplot(
    data=traffic_data_proc,
    x=traffic_data_proc["label"].map({0: "Not Tor", 1: "Tor"}),
    hue=traffic_data_proc["label"].map({0: "Not Tor", 1: "Tor"}),
    palette="Blues",
)
plt.title("Distribution of Network Traffic")
plt.savefig(os.path.join(figures_path, "target-hist.png"))

# Visualizing Flow Duration
fig, ax = plt.subplots(1, ncols=2, figsize=(15, 10))
fig.suptitle("Flow Duration (Hours)")
sns.histplot(
    data=traffic_data_proc,
    x=traffic_data_proc["Flow Duration"].apply(lambda x: x / 3600),
    hue=traffic_data_proc["label"].map({0: "Not Tor", 1: "Tor"}),
    palette="Blues",
    ax=ax[0],
)

sns.violinplot(
    data=traffic_data_proc,
    x=traffic_data_proc["label"].map({0: "Not Tor", 1: "Tor"}),
    y=traffic_data_proc["Flow Duration"].apply(lambda x: x / 3600),
    hue=traffic_data_proc["label"].map({0: "Not Tor", 1: "Tor"}),
    palette="Blues",
    ax=ax[1],
).set(ylabel="Flow Duration (Hours)")
plt.ticklabel_format(style="plain", axis="y")
plt.savefig(os.path.join(figures_path, "flow-duration-dist.png"))
plt.show()

# Visualizing Flow Bytes/s (amount of data transferred) converted to Gigabytes.
sns.violinplot(
    data=traffic_data_proc,
    x=traffic_data_proc["label"].map({0: "Not Tor", 1: "Tor"}),
    y=traffic_data_proc["Flow Bytes/s"].apply(lambda x: np.log(x + 0.001 / 1024**3)),
    hue=traffic_data_proc["label"].map({0: "Not Tor", 1: "Tor"}),
    palette="Blues",
)
plt.title("Log(Flow GB/s)")
plt.ylabel("Log(Flow GB/s)")
plt.ticklabel_format(style="plain", axis="y")
plt.savefig(os.path.join(figures_path, "log-flow-gbs.png"))

# Visualizing Flow Packets/s
sns.violinplot(
    data=traffic_data_proc,
    x=traffic_data_proc["label"].map({0: "Not Tor", 1: "Tor"}),
    y=traffic_data_proc["Flow Packets/s"].apply(lambda x: np.log(x)),
    hue=traffic_data_proc["label"].map({0: "Not Tor", 1: "Tor"}),
    palette="Blues",
)
plt.title("Log(Flow Packets/s)")
plt.ylabel("Log(Flow Packets/s)")
plt.ticklabel_format(style="plain", axis="y")
plt.savefig(os.path.join(figures_path, "log-flow-packets.png"))

# Visualizing the Distribution of Protocol
sns.histplot(
    data=traffic_data_proc,
    x=traffic_data_proc["Protocol"].map({17: "UDP", 6: "TCP"}),
    hue=traffic_data_proc["label"].map({0: "Not Tor", 1: "Tor"}),
    palette="Blues",
)
plt.title("Distribution of Protocol by Label")
plt.savefig(os.path.join(figures_path, "protocol_hist.png"))
