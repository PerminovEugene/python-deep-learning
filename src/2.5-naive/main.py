import tensorflow as tf
# import tf.keras.datasets import mnist

from naive_sequential import NaiveSequential
from naive_dense import NaiveDense
from training import fit

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

model = NaiveSequential([
  NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
  NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
])

assert len(model.weights) == 4

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

fit(model, train_images, test_images, epochs=10, batch_size=128)

# estimate

predictions = model(test_images)
predictions = predictions.numpy()
predicted_labels = tf.np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print(f"accuracy: {matches.mean():.2f}")