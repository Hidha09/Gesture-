import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ================================================================
#  LOAD DATA
# ================================================================
print("Loading data...")
df = pd.read_csv("gesture_data.csv", header=None)

print(f"Total samples: {len(df)}")
print("\nSamples per gesture:")
print(df.iloc[:, -1].value_counts())

X = df.iloc[:, :-1].values   # 63 landmark features
y = df.iloc[:, -1].values    # gesture labels

# ================================================================
#  ENCODE LABELS
# ================================================================
encoder = LabelEncoder()
y_encoded  = encoder.fit_transform(y)
y_onehot   = to_categorical(y_encoded)
num_classes = len(encoder.classes_)

print(f"\nClasses: {list(encoder.classes_)}")

# ================================================================
#  TRAIN / TEST SPLIT
# ================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ================================================================
#  BUILD MODEL
# ================================================================
model = Sequential([
    Dense(256, activation='relu', input_shape=(63,)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================================================================
#  TRAIN
# ================================================================
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint("gesture_model.h5", save_best_only=True, verbose=1)
]

print("\nTraining...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# ================================================================
#  EVALUATE
# ================================================================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# ================================================================
#  PLOT
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'],    label='Train Accuracy')
ax1.plot(history.history['val_accuracy'],label='Val Accuracy')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(history.history['loss'],    label='Train Loss')
ax2.plot(history.history['val_loss'],label='Val Loss')
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()
print("Training graph saved as training_results.png")
print("Model saved as gesture_model.h5")