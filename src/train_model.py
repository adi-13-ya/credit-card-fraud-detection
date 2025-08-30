import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# Load the dataset
df = pd.read_csv("data/creditcard.csv")

# Optional: Shuffle data (important in imbalanced datasets)
df = shuffle(df, random_state=42)

# Separate features and target
X = df.drop("Class", axis=1)  # Features
y = df["Class"]               # Target (0 = legit, 1 = fraud)

# Scale the features (Important for NN convergence)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test (Stratify to keep fraud ratio same)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Data prepared. Training samples:", len(X_train), "Test samples:", len(X_test))

# removing warnings

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide most TF logs

# Suppress protobuf runtime version warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')


from sklearn.utils import class_weight
import numpy as np

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)


import tensorflow as tf
from keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Define the neural network
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_auc', patience=5, mode='max' , restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=2048,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=2,
    class_weight = class_weights
)

model.save("./models/fraud_model.keras")



