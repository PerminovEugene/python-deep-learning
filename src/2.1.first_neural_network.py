from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([ # is a stack of layers. Data passes through them in order.
  layers.Dense(
    512, # amount of neurons in layer
    activation="relu" # Each neuron computes relu(w·x + b), where relu(z) = max(0, z)
  ),
  layers.Dense(
    10, # neuron per digit
    activation="softmax" # turns scores to probabilities
  )
])

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model.compile(
  # rmsprop is the rule used to update the weights from the gradients
  optimizer="rmsprop",
  # measures how wrong the predicted probabilities are. "Sparse" means the labels are plain integers like 7, not
  # one-hot vectors like [0,0,0,0,0,0,0,1,0,0].
  loss="sparse_categorical_crossentropy",

  metrics=["accuracy"]
)


# flattens each 28×28 image into a vector of 784 numbers
train_images = train_images.reshape((60000, 28 * 28))
# scales the pixels from 0–255 down to 0.0–1.0.
# networks train much better on small, normalized inputs.
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# the weights are updated after every 128 images,
# so one pass over the data is 60000 / 128 ≈ 469 steps.
model.fit(
  train_images,
  train_labels,
  epochs=5,
  batch_size=128
)


# particulart predictions
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print(f"Single digit prediction: {predictions[3]}")


# total precision
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")
print(f"test_loss: {test_loss}")

# Questions and answers:
# 1. What is w?

# w is not one number. Each neuron has a whole vector of 784 weights,
# one per pixel. x is the whole image, all 784 pixel values. So w·x is a dot
# product:

# w·x = w[0]*x[0] + w[1]*x[1] + ... + w[783]*x[783]

# You multiply each pixel by its own weight and add everything up.
# The result is one number. Then you add b and apply ReLU.

# Here's a tiny version with 4 pixels instead of 784:
# x = [0.0, 0.5, 1.0, 0.2]      ← the image (pixels)
# w = [0.3, -0.1, 0.4, 0.2]     ← this neuron's weights
# b = 0.05                      ← this neuron's bias

# w·x = 0.3*0.0 + (-0.1)*0.5 + 0.4*1.0 + 0.2*0.2
#     = 0 - 0.05 + 0.4 + 0.04 = 0.39
# w·x + b = 0.44
# relu(0.44) = 0.44             ← the neuron's output: ONE number

# 2. What are the initial values?

# They're random, but in a much smaller range than -1 to 1. By default Keras uses "Glorot uniform," which picks from [-limit, +limit], where limit =
# sqrt(6 / (inputs + outputs)):
# - Layer 1: sqrt(6 / (784 + 512)) ≈ ±0.068
# - Layer 2: sqrt(6 / (512 + 10)) ≈ ±0.107

# The range is small so that summing 784 products doesn't produce huge numbers at the start.

# Biases start at 0.

# 3. How do they change?

# They change after every batch, not every epoch. With 469 batches per epoch and 5 epochs, that's about 2,345 updates.

# Each update applies this to every weight and every bias:
# w_new = w_old - learning_rate * gradient
# - The gradient for a weight says how much the loss would change if you increased that weight a little. Backpropagation computes it.
# - If the gradient is positive (increasing the weight makes things worse), the weight decreases, and vice versa.
# - learning_rate is a small step size. The default for rmsprop is 0.001. RMSprop also scales each weight's step based on its recent gradients, but the
#   idea is the same.

# 4. Is the bias fixed?

# No. The bias is learned exactly like the weights. It starts at 0 and gets updated after every batch. It has no fixed range and ends up wherever
# training pushes it, usually small values like -0.3 to 0.3. Each neuron has one bias.

# 5. Does a neuron return one value or a vector?

# It returns one number per image.

# 6. Are neurons in the same layer connected to each other?

# No. Neurons in the same layer don't know about each other. Each one:
# - receives all outputs from the previous layer (or all pixels, for layer 1),
# - has its own weights and its own bias,
# - computes its number independently.

# The only connections go between layers, from each neuron in one layer to every neuron in the next.

# 7. Does layer 1 return 512 values or 512 vectors?

# It returns 512 numbers, which together form one vector of length 512 per image. That vector is the input x for layer 2. Each of the 10 neurons in
# layer 2 has its own 512 weights, does the same relu-less w·x + b, and returns one number. Softmax then turns those 10 numbers into 10 probabilities. 
