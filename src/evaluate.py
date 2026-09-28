import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

test = np.load("data/processed/test.npz")
model = tf.keras.models.load_model("models/model.h5")

loss, acc = model.evaluate(test["x"], test["y"], verbose=0)
preds = model.predict(test["x"], verbose=0).argmax(axis=1)

os.makedirs("reports", exist_ok=True)
cm = confusion_matrix(test["y"], preds)
fig, ax = plt.subplots(figsize=(9, 9))
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
    ax=ax, xticks_rotation=45, colorbar=False)
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png")

with open("metrics.json", "w") as f:
    json.dump({"test_loss": round(float(loss), 4), "test_accuracy": round(float(acc), 4)}, f, indent=2)

print(f"Test accuracy: {acc:.4f} | Test loss: {loss:.4f}")