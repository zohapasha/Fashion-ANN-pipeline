import os

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["preprocess"]

# I am making a change to check git stash and checkout again since previous git messed up :()

def normalize(x):
    # Standardize using the Fashion-MNIST mean and standard deviation
    return (x.astype("float32") / 255.0 - 0.2860) / 0.3530


train = np.load("data/raw/train.npz")
test = np.load("data/raw/test.npz")

x_train, y_train = normalize(train["x"]), train["y"]
x_test, y_test = normalize(test["x"]), test["y"]

x_tr, x_val, y_tr, y_val = train_test_split(
    x_train, y_train,
    test_size=params["test_size"],
    random_state=params["seed"],
    stratify=y_train,
)

os.makedirs("data/processed", exist_ok=True)
np.savez_compressed("data/processed/train.npz", x=x_tr, y=y_tr)
np.savez_compressed("data/processed/val.npz", x=x_val, y=y_val)
np.savez_compressed("data/processed/test.npz", x=x_test, y=y_test)

print(f"train {x_tr.shape} | val {x_val.shape} | test {x_test.shape}")