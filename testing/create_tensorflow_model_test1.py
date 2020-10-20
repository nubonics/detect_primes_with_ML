import pandas as pd
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow.keras import layers

csv_path = '/home/nubonix/PycharmProjects/detect_primes_with_ML/is_prime.csv'
df = pd.read_csv(csv_path)
train_features = df['integer'].astype(float)[33_000:]
train_labels = df['is_prime'].astype(bool)[33_000:]
test_features = df['integer'].astype(float)[:33_000]
test_labels = df['is_prime'].astype(bool)[:33_000]

norm_is_prime_model = tf.keras.Sequential([
    # layers.Dense(10),
    layers.Dense(1),
])

norm_is_prime_model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# norm_is_prime_model.fit(train_features, train_labels, epochs=1, validation_data=(test_features, test_labels))
norm_is_prime_model.fit(train_features, train_labels, epochs=3)

# norm_is_prime_model.evaluate(test_features, test_labels, batch_size=32)
norm_is_prime_model.evaluate(test_features, test_labels)

norm_is_prime_model.save('../my_model.h5')
