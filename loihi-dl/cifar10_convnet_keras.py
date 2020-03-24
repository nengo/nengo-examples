import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as klayers

relu = "relu"

input_shape = (32, 32, 3)

layer_confs = [
    dict(n_filters=4, kernel_size=1, strides=1, neuron_type=relu, on_chip=False),
    dict(n_filters=64, kernel_size=3, strides=2, neuron_type=relu, on_chip=True),
    dict(n_filters=96, kernel_size=3, strides=1, neuron_type=relu, on_chip=True),
    dict(n_filters=128, kernel_size=3, strides=2, neuron_type=relu, on_chip=True),
    dict(n_filters=128, kernel_size=1, strides=1, neuron_type=relu, on_chip=True),
    dict(n_neurons=20, neuron_type=relu, on_chip=True),
    dict(n_neurons=10, neuron_type=None, on_chip=False),
]

inp = klayers.Input(shape=input_shape, name="input")

layers = []
layer_outputs = []
x = inp
for k, layer_conf in enumerate(layer_confs):
    neuron_type = layer_conf.pop("neuron_type")
    on_chip = layer_conf.pop("on_chip", True)
    name = layer_conf.pop("name", "layer%d" % k)

    if "n_filters" in layer_conf:
        # convolutional layer
        n_filters = layer_conf.pop("n_filters")
        kernel_size = layer_conf.pop("kernel_size")
        strides = layer_conf.pop("strides", 1)
        assert len(layer_conf) == 0, "Unused fields in conv layer: %s" % list(
            layer_conf
        )

        layer = klayers.Conv2D(
            n_filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=neuron_type,
            padding="valid",
            name=name,
        )
        y = layer(x)
    else:
        # dense layer
        n_neurons = layer_conf.pop("n_neurons")

        y = klayers.Flatten()(x)
        layer = klayers.Dense(n_neurons, activation=neuron_type, name=name)
        y = layer(y)

    layers.append(layer)
    layer_outputs.append(y)
    x = y

layers_model = keras.Model([inp], layer_outputs)
model = keras.Model([inp], [y])

# --- load dataset
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
n_classes = 10
train_t = np.array(tf.one_hot(train_y, n_classes), dtype=np.float32)
test_t = np.array(tf.one_hot(test_y, n_classes), dtype=np.float32)
train_t = np.squeeze(train_t, axis=1)
test_t = np.squeeze(test_t, axis=1)

train_x = train_x.astype(np.float32) / 127.5 - 1
test_x = test_x.astype(np.float32) / 127.5 - 1

assert train_x[0].shape == input_shape

# train_x_flat = train_x.reshape((train_x.shape[0], -1))
# train_t_flat = train_t.reshape((train_t.shape[0], -1))

# test_x_flat = test_x.reshape((test_x.shape[0], -1))
# test_t_flat = test_t.reshape((test_t.shape[0], -1))

# --- evaluate on test data
for k, layer in enumerate(layers):
    weights = layer.get_weights()[0]
    print("Layer %d weights: %0.3f" % (k, np.abs(weights).mean()))

outputs = layers_model.predict(test_x[:100])
for k, out in enumerate(outputs):
    print("Layer %d: %0.3f" % (k, np.mean(out)))

# --- train network in NengoDL
checkpoint_base = "./cifar10_convnet_params"

batch_size = 256

# train_idg = tf.keras.preprocessing.image.ImageDataGenerator(
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=20,
#     # shear_range=0.1,
#     horizontal_flip=True,
# )
# train_idg.fit(train_x)
# train_idg_flow = train_idg.flow(train_x, train_y, batch_size=batch_size)

# def train_generator():
#     # yield (50000 // batch_size) * np.ones((batch_size, 1))

#     for x, y in train_idg_flow:
#         x = tf.reshape(x, (x.shape[0], 1, -1))
#         y = tf.reshape(y, (y.shape[0], 1, -1))
#         yield (x, y)

model.compile(
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=tf.optimizers.Adam(),
    metrics=["accuracy"],
)

# run training
# model.fit(train_x, train_t, epochs=10)
model.fit(train_x, train_t, epochs=30)
