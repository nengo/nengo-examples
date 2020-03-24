from functools import partial
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import nengo
import nengo_dl
import nengo_loihi

max_rate = 1
amp = 1.0

relu = nengo.RectifiedLinear(amplitude=amp)

layer_confs = [
    dict(n_filters=4, kernel_size=1, strides=1, neuron_type=relu, on_chip=False),
    dict(n_filters=64, kernel_size=3, strides=2, neuron_type=relu, on_chip=True),
    dict(n_filters=96, kernel_size=3, strides=1, neuron_type=relu, on_chip=True),
    dict(n_filters=128, kernel_size=3, strides=2, neuron_type=relu, on_chip=True),
    dict(n_filters=128, kernel_size=1, strides=1, neuron_type=relu, on_chip=True),
    dict(n_neurons=20, neuron_type=relu, on_chip=True),
    dict(n_neurons=10, neuron_type=None, on_chip=False),
]

input_shape = nengo.transforms.ChannelShape((32, 32, 3), channels_last=True)

with nengo.Network(seed=0) as net:
    # net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
    # net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])

    net.config[nengo.Ensemble].gain = nengo.dists.Choice([1.0])
    net.config[nengo.Ensemble].bias = nengo.dists.Choice([0.0])

    net.config[nengo.Connection].synapse = None

    # this is an optimization to improve the training speed,
    # since we won't require stateful behaviour in this example
    # (copied from MNIST example)
    nengo_dl.configure_settings(stateful=False)

    nengo_loihi.add_params(net)  # allow setting on_chip

    # the input node that will be used to feed in input images
    inp = nengo.Node([0] * input_shape.size, label="input")

    connections = []
    transforms = []
    layer_probes = []
    shape_in = input_shape
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

            kernel_size = (
                (kernel_size, kernel_size)
                if isinstance(kernel_size, int)
                else kernel_size
            )
            strides = (strides, strides) if isinstance(strides, int) else strides

            transform = nengo.Convolution(
                n_filters=n_filters,
                input_shape=shape_in,
                kernel_size=kernel_size,
                strides=strides,
                padding="valid",
                init=nengo_dl.dists.Glorot(scale=1. / np.prod(kernel_size)),
            )
            shape_out = transform.output_shape
        else:
            # dense layer
            n_neurons = layer_conf.pop("n_neurons")
            assert len(layer_conf) == 0, "Unused fields in dense layer: %s" % list(
                layer_conf
            )

            shape_out = nengo.transforms.ChannelShape((n_neurons,))
            transform = nengo.Dense(
                (shape_out.size, shape_in.size), init=nengo_dl.dists.Glorot(),
            )

        if neuron_type is None:
            assert not on_chip, "Nodes can only be run off-chip"
            y = nengo.Node(size_in=shape_out.size, label=name)
        else:
            ens = nengo.Ensemble(shape_out.size, 1, neuron_type=neuron_type, label=name)
            net.config[ens].on_chip = on_chip
            y = ens.neurons

        conn = nengo.Connection(x, y, transform=transform, synapse=None)

        transforms.append(transform)
        connections.append(conn)
        x = y
        shape_in = shape_out

    output_p = nengo.Probe(x, synapse=None, label="output_p")

# --- load dataset
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
n_classes = 10
train_t = np.array(tf.one_hot(train_y, n_classes), dtype=np.float32)
test_t = np.array(tf.one_hot(test_y, n_classes), dtype=np.float32)

train_x = train_x.astype(np.float32) / 127.5 - 1
test_x = test_x.astype(np.float32) / 127.5 - 1

assert train_x[0].shape == input_shape.shape

# train_x_flat = train_x.reshape((train_x.shape[0], 1, -1))
# train_t_flat = train_t.reshape((train_t.shape[0], 1, -1))
train_x_flat = train_x.reshape((train_x.shape[0], 1, -1))
train_t_flat = train_t.reshape((train_t.shape[0], 1, -1))

test_x_flat = test_x.reshape((test_x.shape[0], 1, -1))
test_t_flat = test_t.reshape((test_t.shape[0], 1, -1))

# --- evaluate layers
with nengo_dl.Simulator(net, minibatch_size=100, progress_bar=False) as sim:
    for k, conn in enumerate(connections):
        weights = sim.model.sig[conn]["weights"].initial_value
        print("Layer %d initial weights: %0.3f" % (k, np.abs(weights).mean()))

    sim.run_steps(1, data={inp: train_x_flat[:100]})

for k, layer_probe in enumerate(layer_probes):
    out = sim.data[layer_probe][-1]
    print("Layer %d initial rates: %0.3f" % (k, np.mean(out)))

# --- train network in NengoDL
batch_size = 256

with nengo_dl.Simulator(net, minibatch_size=batch_size) as sim:
    losses = tf.losses.CategoricalCrossentropy(from_logits=True)
    metrics = ["accuracy"]

    sim.compile(
        loss=losses,
        optimizer=tf.optimizers.Adam(),
        # optimizer=tf.optimizers.RMSprop(0.001),
        metrics=metrics,
    )

    # --- train
    train_inputs = {inp: train_x_flat}
    train_targets = train_t_flat

    # sim.fit(x=train_inputs, y=train_targets, epochs=1)
    sim.fit(x=train_inputs, y=train_targets, epochs=10)
