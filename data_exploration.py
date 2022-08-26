import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import re

model_data = pd.read_csv(
    "Data till AFS (antal brott).csv"
)

model_data = model_data.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
model_data = model_data[model_data["oro"].notna()]

test_data = model_data[model_data["ntu_year"] == 2021]
train_val_data = model_data[model_data["ntu_year"] != 2021]


def split_to_feature_and_target(df):
    features = df.drop(["name", "oro", "lpo"], axis=1)
    target = df.oro
    features = features.replace(to_replace=r",", value=".", regex=True)
    features = features.astype(float)
    target = target.replace(to_replace=r",", value=".", regex=True)
    target = target.astype(float)
    return features, target


features, target = split_to_feature_and_target(train_val_data)
x_train, x_val, y_train, y_val = train_test_split(
    features, target, test_size=0.25, random_state=42
)

model = lgb.LGBMRegressor(
    num_leaves=37,
    boosting="gbdt",
    n_estimators=23,
    learning_rate=0.106,
    seed=100,
    loss="mean_squared_error",
)
model.fit(
    x_train,
    y_train,
    eval_set=[(x_train, y_train), (x_val, y_val)],
    verbose=20,
    eval_metric="mean_squared_error",
)

features_test, target_test = split_to_feature_and_target(test_data)
predictions = model.predict(features_test)


data_to_plot = test_data[["lpo", "oro"]]
data_to_plot["prediction"] = predictions

fig, ax = plt.subplots(figsize=(12, 8))
colormap = cm.viridis
colorlist = [
    colors.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, len(data_to_plot["lpo"]))
]
for i, c in enumerate(colorlist):
    x = data_to_plot["oro"].iloc[i]
    y = data_to_plot["prediction"].iloc[i]
    l = data_to_plot["lpo"].iloc[i]
    ax.scatter(x, y, label=l, s=50, linewidth=0.1, c=c)
ax.legend(fontsize=3, loc="center left", bbox_to_anchor=(1, 0.5), ncol=3)
plt.xlabel("NTU worry score (oro)")
plt.ylabel("model predictions")
plt.title("Scatter plot showing NTU worry score vs. model predictions")
fig.tight_layout()
plt.savefig('Scatter_plot.png')
