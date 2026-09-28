from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
  layers.Dense(512, activation="relu"),
  layers.Dense(10, activation="softmax")
])

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuract"])


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

model.fit(train_images, train_labels, epoch=5, batch_size=128)


# particulart predictions
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print(f"Single digit prediction: {predictions[3]}")


# total precision
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")
print(f"test_loss: {test_loss}")