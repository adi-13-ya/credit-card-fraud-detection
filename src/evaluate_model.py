import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide most TF logs

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.utils import shuffle

# Load dataset
df = pd.read_csv("data/creditcard.csv")
df = shuffle(df, random_state=42)

# Prepare features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Load the trained model
model = tf.keras.models.load_model("models/fraud_model.keras")

# Evaluate the model
test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nModel Evaluation:\n-----------------")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predictions
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs >= 0.7).astype("int")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("\nClassification Report:\n-----------------------")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], digits=4))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
