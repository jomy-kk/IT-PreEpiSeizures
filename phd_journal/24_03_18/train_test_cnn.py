from datetime import timedelta
from os.path import exists

import keras
import numpy as np
from sklearn.model_selection import train_test_split

from ltbio.processing.formaters import Normalizer, Segmenter
from read import *


SEG_LEN = 5  # seconds
objects_filename = f"kjpp_objects_{SEG_LEN}s.npy"
targets_filename = f"kjpp_targets_{SEG_LEN}s.npy"

if exists(objects_filename) and exists(targets_filename):
    objects = np.load(objects_filename)
    targets = np.load(targets_filename)
else:
    # 1) Get all EEG signals (with ages)
    signals = read_all_eeg('KJPP', N=10)

    # 2) Normalise and segment signals and associate targets
    normalizer = Normalizer(method='minmax')
    segmenter = Segmenter(timedelta(seconds=5))
    objects, targets = [], []
    for i, signal in enumerate(signals):
        if i % 100 == 0:
            print(f"Processed {i/len(signals)*100:.2f}% of signals")
        signal = normalizer(signal)
        signal = segmenter(signal)
        age = signal._Biosignal__patient._Patient__age / 12 /12  # Divide targets by 12 to get age in years
        n_segments = len(signal['T5'].subdomains)
        for j in range(n_segments):
            v_block = signal._vblock(j)
            if v_block.shape == (20, 640):  # Remove objects that do not have the shape (20, 640)
                objects.append(v_block)
                targets.append(age)

    assert len(objects) == len(targets)

    # Save objects and targets
    np.save(objects_filename, objects)
    np.save(targets_filename, targets)

    del signals

exit(0)


# 3) Define model
"""
Architecture:
    Intput Layer: (20, 640)
    Convolutional Layer: Kernel (7, 128), Stride (1, 3), Padding 'same'
    Convolutional Layer: Kernel (7, 64), Stride (1, 3), Padding 'same'
    Convolutional Layer: Kernel (7, 32), Stride (1, 3), Padding 'same'
    Convolutional Layer: Kernel (7, 16), Stride (1, 3), Padding 'same'
    Global Average Pooling Layer: (16,)
    Linear Layer: (16, 1)
"""
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(20, 640)),
    keras.layers.Reshape((20, 640, 1)),
    keras.layers.Conv1D(16, 7, strides=(1, 3), padding='same', activation='sigmoid'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv1D(32, 7, strides=(1, 3), padding='same', activation='sigmoid'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv1D(64, 7, strides=(1, 3), padding='same', activation='sigmoid'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv1D(128, 7, strides=(1, 3), padding='same', activation='sigmoid'),
    keras.layers.BatchNormalization(),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(1)
])
model.summary()

# 4. Train settings
# Adam with starting learning rate of 3e-4, with reduce on plateau with patience of 3 epochs
model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4),
              loss=keras.losses.MeanAbsoluteError(),
              # metrics = MAE, MSE, R2
              metrics=[keras.metrics.MeanAbsoluteError(),
                       keras.metrics.MeanSquaredError(),
                       keras.metrics.R2Score()])

training_conditions = {
    'batch_size': 512,
    'epochs': 100,
    'validation_split': 0.2,
    'callbacks': [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    ]
}

# 4) Train and test model
# Split dataset
dataset = list(zip(objects, targets))
train, test = train_test_split(dataset, test_size=0.2)
train_objects, train_targets = zip(*train)
test_objects, test_targets = zip(*test)
# make numpy arrays
train_objects = np.array(train_objects)
test_objects = np.array(test_objects)
train_targets = np.array(train_targets)
test_targets = np.array(test_targets)

# Train model
model.fit(train_objects, train_targets, **training_conditions)

# Test model
loss, mae, mse, r2 = model.evaluate(test_objects, test_targets)
print(f"Test loss: {loss}")
print(f"Test MAE: {mae}")
print(f"Test MSE: {mse}")
print(f"Test R2: {r2}")

# Plot predictions
import matplotlib.pyplot as plt
predictions = model.predict(test_objects)
plt.scatter(test_targets, predictions)
plt.xlabel('True Age')
plt.ylabel('Predicted Age')
plt.show()

