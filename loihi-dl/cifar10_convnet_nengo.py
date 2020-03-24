import collections
from functools import partial
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import nengo
import nengo_dl
import nengo_loihi


class NengoImageIterator(tf.keras.preprocessing.image.Iterator):
    def __init__(
        self,
        image_data_generator,
        x_keys,
        x,
        y_keys,
        y,
        batch_size=32,
        shuffle=False,
        sample_weight=None,
        seed=None,
        subset=None,
        dtype="float32",
    ):
        assert subset is None, "Not Implemented"
        assert isinstance(x_keys, (tuple, list))
        assert isinstance(y_keys, (tuple, list))
        assert isinstance(x, (tuple, list))
        assert isinstance(y, (tuple, list))

        self.dtype = dtype
        self.x_keys = x_keys
        self.y_keys = y_keys

        x0 = x[0]
        assert all(len(xx) == len(x0) for xx in x), (
            "All of the arrays in `x` should have the same length. "
            "Found a pair with: len(x[0]) = %s, len(x[?]) = %s" % (len(x0), len(xx))
        )
        assert all(len(yy) == len(x0) for yy in y), (
            "All of the arrays in `y` should have the same length as `x`. "
            "Found one with: len(x[0]) = %s, len(y[?]) = %s" % (len(x0), len(yy))
        )
        assert len(x_keys) == len(x)
        assert len(y_keys) == len(y)

        if sample_weight is not None and len(x0) != len(sample_weight):
            raise ValueError(
                "`x[0]` (images tensor) and `sample_weight` "
                "should have the same length. "
                "Found: x.shape = %s, sample_weight.shape = %s"
                % (np.asarray(x0).shape, np.asarray(sample_weight).shape)
            )

        self.x = [
            np.asarray(xx, dtype=self.dtype if i == 0 else None)
            for i, xx in enumerate(x)
        ]
        if self.x[0].ndim != 4:
            raise ValueError(
                "Input data in `NumpyArrayIterator` "
                "should have rank 4. You passed an array "
                "with shape",
                self.x[0].shape,
            )

        self.y = [np.asarray(yy) for yy in y]
        self.sample_weight = (
            None if sample_weight is None else np.asarray(sample_weight)
        )
        self.image_data_generator = image_data_generator
        super().__init__(self.x[0].shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        images = self.x[0]
        assert images.dtype == self.dtype

        n = len(index_array)
        batch_x = np.zeros((n,) + images[0].shape, dtype=self.dtype)
        for i, j in enumerate(index_array):
            x = images[j]
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(x, params)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

        batch_x_miscs = [xx[index_array] for xx in self.x[1:]]
        batch_y_miscs = [yy[index_array] for yy in self.y]

        x_pairs = [
            (k, self.x_postprocess(k, v))
            for k, v in zip(self.x_keys, [batch_x] + batch_x_miscs)
        ]
        y_pairs = [
            (k, self.y_postprocess(k, v)) for k, v in zip(self.y_keys, batch_y_miscs)
        ]

        output = (
            collections.OrderedDict(x_pairs),
            collections.OrderedDict(y_pairs),
        )

        if self.sample_weight is not None:
            output += (self.sample_weight[index_array],)
        return output

    def x_postprocess(self, key, x):
        return x if key == "n_steps" else x.reshape((x.shape[0], 1, -1))

    def y_postprocess(self, key, y):
        return y.reshape((y.shape[0], 1, -1))


def percentile_l2_loss(
    y_true, y, sample_weight=None, weight=1.0, target=0.0, percentile=99.0
):
    # y axes are (batch examples, time (==1), neurons)
    assert len(y.shape) == 3
    rates = tfp.stats.percentile(y, percentile, axis=(0, 1))
    loss = tf.nn.l2_loss(rates - target)
    if sample_weight is not None:
        weight = weight * sample_weight
    return (weight * loss) if weight != 1.0 else loss


def percentile_l2_loss_range(
    y_true, y, sample_weight=None, weight=1.0, min=0.0, max=np.inf, percentile=99.0
):
    # y axes are (batch examples, time (==1), neurons)
    assert len(y.shape) == 3
    rates = tfp.stats.percentile(y, percentile, axis=(0, 1))
    low_error = tf.maximum(0.0, min - rates)
    high_error = tf.maximum(0.0, rates - max)
    loss = tf.nn.l2_loss(low_error + high_error)
    if sample_weight is not None:
        weight = weight * sample_weight
    return (weight * loss) if weight != 1.0 else loss


def has_checkpoint(checkpoint_base):
    checkpoint_dir, checkpoint_name = os.path.split(checkpoint_base)
    if not os.path.exists(checkpoint_dir):
        return False

    files = os.listdir(checkpoint_dir)
    files = [f for f in files if f.startswith(checkpoint_name)]
    return len(files) > 0


max_rate = 150
amp = 1.0 / max_rate
# rate_reg = 1e-2
rate_reg = 1e-3
rate_target = max_rate * amp  # must be in amplitude scaled units

# max_rate = 1
# amp = 1.0

lif = nengo.LIF(amplitude=amp)
relu = nengo.SpikingRectifiedLinear(amplitude=amp)
# lif = nengo_dl.neurons.SoftLIFRate(amplitude=amp, sigma=0.01)
# relu = nengo.RectifiedLinear(amplitude=amp)

layer_confs = [
    dict(n_filters=4, kernel_size=1, strides=1, neuron_type=relu, on_chip=False),
    dict(n_filters=64, kernel_size=3, strides=2, neuron_type=lif, on_chip=True),
    dict(n_filters=96, kernel_size=3, strides=1, neuron_type=lif, on_chip=True),
    dict(n_filters=128, kernel_size=3, strides=2, neuron_type=lif, on_chip=True),
    dict(n_filters=128, kernel_size=1, strides=1, neuron_type=lif, on_chip=True),
    # dict(n_filters=64, kernel_size=3, strides=2, neuron_type=relu, on_chip=True),
    # dict(n_filters=96, kernel_size=3, strides=1, neuron_type=relu, on_chip=True),
    # dict(n_filters=128, kernel_size=3, strides=2, neuron_type=relu, on_chip=True),
    # dict(n_filters=128, kernel_size=1, strides=1, neuron_type=relu, on_chip=True),
    dict(n_neurons=20, neuron_type=lif, on_chip=True),
    # dict(n_neurons=20, neuron_type=relu, on_chip=True),
    dict(n_neurons=10, neuron_type=None, on_chip=False),
]

# layer_confs = [
#     dict(n_filters=4, kernel_size=1, strides=1, neuron_type=relu, on_chip=False),
#     dict(n_neurons=10, neuron_type=None, on_chip=False),
# ]

input_shape = nengo.transforms.ChannelShape((32, 32, 3), channels_last=True)

with nengo.Network() as net:
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    # net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([-0.01])

    # net.config[nengo.Ensemble].gain = nengo.dists.Choice([1.0])
    # net.config[nengo.Ensemble].bias = nengo.dists.Choice([0.0])

    net.config[nengo.Connection].synapse = None

    nengo_dl.configure_settings(keep_history=True)

    # this is an optimization to improve the training speed,
    # since we won't require stateful behaviour in this example
    # (copied from MNIST example)
    nengo_dl.configure_settings(stateful=False)

    nengo_dl.configure_settings(lif_smoothing=0.01)
    # nengo_dl.configure_settings(lif_smoothing=0.2)

    nengo_loihi.add_params(net)  # allow setting on_chip

    # the input node that will be used to feed in input images
    inp = nengo.Node([0] * input_shape.size, label="input_node")

    connections = []
    transforms = []
    layer_probes = []
    shape_in = input_shape
    x = inp
    for k, layer_conf in enumerate(layer_confs):
        neuron_type = layer_conf.pop("neuron_type")
        on_chip = layer_conf.pop("on_chip", True)
        name = layer_conf.pop("name", "layer%d" % k)

        # --- create layer transform
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
                init=nengo_dl.dists.Glorot(scale=1.0 / np.prod(kernel_size)),
            )
            shape_out = transform.output_shape

            n_weights = np.prod(transform.kernel_shape)
            print(
                "%s: conv %s, stride %s, output %s (%d weights)"
                % (name, kernel_size, strides, shape_out.shape, n_weights)
            )
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

            print(
                "%s: dense %d, output %s (%d weights)"
                % (name, n_neurons, shape_out.shape, np.prod(transform.shape))
            )

        # --- create layer output (Ensemble or Node)
        if neuron_type is None:
            assert not on_chip, "Nodes can only be run off-chip"
            y = nengo.Node(size_in=shape_out.size, label=name)
        else:
            ens = nengo.Ensemble(shape_out.size, 1, neuron_type=neuron_type, label=name)
            net.config[ens].on_chip = on_chip
            y = ens.neurons

            # add a probe so we can measure individual layer rates
            probe = nengo.Probe(y, synapse=None, label="%s_p" % name)
            net.config[probe].keep_history = False
            layer_probes.append(probe)

        conn = nengo.Connection(x, y, transform=transform)

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

assert train_x[0].shape == test_x[0].shape == input_shape.shape

train_x_flat = train_x.reshape((train_x.shape[0], 1, -1))
train_t_flat = train_t.reshape((train_t.shape[0], 1, -1))

test_x_flat = test_x.reshape((test_x.shape[0], 1, -1))
test_t_flat = test_t.reshape((test_t.shape[0], 1, -1))

# --- evaluate layers
# use rate neurons always by setting learning_phase_scope
with tf.keras.backend.learning_phase_scope(1), nengo_dl.Simulator(
    net, minibatch_size=100, progress_bar=False
) as sim:
    for k, conn in enumerate(connections):
        weights = sim.model.sig[conn]["weights"].initial_value
        print("Layer %d initial weights: %0.3f" % (k, np.abs(weights).mean()))

    sim.run_steps(1, data={inp: train_x_flat[:100]})

for k, layer_probe in enumerate(layer_probes):
    out = sim.data[layer_probe][-1]
    print("Layer %d initial rates: %0.3f" % (k, np.mean(out)))

# --- train network in NengoDL
checkpoint_base = "./cifar10_convnet_nengo_params"

batch_size = 256

train_idg = tf.keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=20,
    # shear_range=0.1,
    horizontal_flip=True,
)
train_idg.fit(train_x)

# use rate neurons always by setting learning_phase_scope
with tf.keras.backend.learning_phase_scope(1), nengo_dl.Simulator(
    net, minibatch_size=batch_size
) as sim:
    if 0:
        # if has_checkpoint(checkpoint_base):
        sim.load_params(checkpoint_base)
    else:

        def dummy_loss(y_true, y_pred):
            return 1.1

        def rate_metric(_, outputs):
            return outputs
            # print(_.shape)
            # print(outputs.shape)
            # return outputs.mean()

        losses = {output_p: tf.losses.CategoricalCrossentropy(from_logits=True)}
        metrics = {output_p: "accuracy"}

        for probe, layer_conf in zip(layer_probes, layer_confs):
            metrics[probe] = rate_metric
            # losses[probe] = dummy_loss

            # losses[probe] = partial(
            #     percentile_l2_loss,
            #     weight=rate_reg,
            #     target=rate_target,
            #     percentile=99.9,
            # )

            if layer_conf.get("on_chip", True):
                losses[probe] = partial(
                    percentile_l2_loss_range,
                    weight=rate_reg,
                    min=0.5 * rate_target,
                    max=rate_target,
                    percentile=99.9,
                )
            else:
                losses[probe] = partial(
                    percentile_l2_loss_range,
                    weight=10 * rate_reg,
                    min=0,
                    max=rate_target,
                    percentile=99.9,
                )

        sim.compile(
            loss=losses,
            # loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.optimizers.Adam(),
            # optimizer=tf.optimizers.RMSprop(0.001),
            metrics=metrics,
        )

        # --- run on test data
        test_inputs = {inp: test_x_flat}
        test_targets = {output_p: test_t_flat}
        for probe in layer_probes:
            test_targets[probe] = np.zeros(
                (test_t_flat.shape[0], 1, 0), dtype=np.float32
            )

        # --- train
        steps_per_epoch = len(train_x) // batch_size
        n = steps_per_epoch * batch_size
        n_steps = np.ones((n, 1), dtype=np.int32)
        train_data = NengoImageIterator(
            image_data_generator=train_idg,
            x_keys=[inp.label, "n_steps"],
            x=[train_x[:n], n_steps],
            y_keys=[output_p.label] + [probe.label for probe in layer_probes],
            y=[train_t[:n]]
            + [np.zeros((n, 1, 0), dtype=np.float32) for _ in layer_probes],
            batch_size=batch_size,
            shuffle=True,
        )

        # n_epochs = 30
        n_epochs = 200

        for epoch in range(n_epochs):
            sim.fit(
                train_data, steps_per_epoch=steps_per_epoch, epochs=1, verbose=2,
            )

            outputs = sim.evaluate(x=test_inputs, y=test_targets, verbose=0)
            print("Epoch %d test: %s" % (epoch, outputs))

            # print("Test error after training: %.2f%%" %
            #       sim.loss(test_inputs, test_targets, classification_error))

            # savefile = "%s_%d" % (checkpoint_base, epoch)
            savefile = checkpoint_base
            sim.save_params(savefile)
            print("Saved params to %r" % savefile)

    try:
        train_outputs = sim.evaluate(x=train_inputs, y=train_targets, verbose=0)
        print("Final train: %s" % (epoch, train_outputs))

        test_outputs = sim.evaluate(x=test_inputs, y=test_targets, verbose=0)
        print("Final test: %s" % (epoch, test_outputs))
    except Exception as e:
        print("Could not compute ANN values on this machine: %s" % e)
