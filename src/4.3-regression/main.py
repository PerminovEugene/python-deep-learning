from tensorflow.keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = (boston_housing.load_data())

# normalization
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

from tensorflow import keras
from tensorflow.keras import layers

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)
  ])
  model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
  return model

import numpy as np
# k-fold cross validation
k = 4
num_value_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
  print(f"Processing fold #{i}")
  val_data = train_data[i * num_value_samples: (i + 1) * num_value_samples]
  val_targets = train_targets[i * num_value_samples: (i + 1) * num_value_samples]

  partial_train_data = np.concatenate(
    [train_data[:i * num_value_samples],
     train_data[(i + 1) * num_value_samples:]],
     axis=0
  )

  partial_train_targets = np.concatenate(
    [train_targets[:i * num_value_samples],
     train_targets[(i +1) * num_value_samples:]],
     axis=0
  )

  model = build_model()
  model.fit(partial_train_data, partial_train_targets,
            epochs=num_epochs, batch_size=16, verbose=0)
  val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
  all_scores.append(val_mae)
print(f"Mean validation MAE: {np.mean(all_scores)}")

num_epochs=500
all_mae_histories = []
for i in range(k):
  print(f"Processing fold #{i}")
  val_data = train_data[i * num_value_samples: (i + 1) * num_value_samples]
  val_targets = train_targets[i * num_value_samples: (i + 1) * num_value_samples]
  partial_train_data = np.concatenate(
    [
      train_data[:i * num_value_samples],
      train_data[(i + 1) * num_value_samples:]],
      axis=0
  )
  partial_train_targets = np.concatenate(
    [
      train_targets[:i * num_value_samples],
      train_targets[(i + 1) * num_value_samples:]],
      axis=0
  )
  model = build_model()
  history = model.fit(partial_train_data, partial_train_targets,
                      validation_data=(val_data, val_targets),
                      epochs=num_epochs, batch_size=16, verbose=0
                      )
  mae_history = history.history["val_mae"]
  all_mae_histories.append(mae_history)

average_mae_history = [
  np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
]

import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Estimation of MAE")
plt.show()

model = build_model()
model.fit(train_data, train_targets, epochs=130, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(f"Test MAE: {test_mae_score}")

