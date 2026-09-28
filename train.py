import os

import numpy as np
import tensorflow as tf
import yaml

p = yaml.safe_load(open("params.yaml"))["train"]
tf.random.set_seed(p["seed"])

train = np.load("data/processed/train.npz")
val = np.load("data/processed/val.npz")

model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(p["dense_units"], activation="relu"),
    tf.keras.layers.Dropout(p["dropout_rate"]),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=p["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

os.makedirs("models", exist_ok=True)
model.fit(
    train["x"], train["y"],
    validation_data=(val["x"], val["y"]),
    epochs=p["epochs"],
    batch_size=p["batch_size"],
    callbacks=[tf.keras.callbacks.CSVLogger("models/history.csv")],
)

model.save("models/model.h5")
print("Saved models/model.h5 and models/history.csv")

# Hello, I have made this change to test git diff with unstaged changes