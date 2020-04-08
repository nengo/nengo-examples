import collections
from functools import partial
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import nengo
import nengo_dl
import nengo_loihi

# --- load dataset
channels_last = True

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

# test_x = test_x[:10]
# test_x = test_x[10:20]
# test_x = test_x[2:6]
# test_x = test_x[2:4]
# test_x = test_x[0:4]

# --- this input causes problems
test_x = test_x[0:1]

# --- this input does not cause problems
# test_x = test_x[1:2]

# q = 10
# q = 16
# q = 17
# test_x = test_x[:, :q, :q, :]

if not channels_last:
    test_x = np.transpose(test_x, (0, 3, 1, 2))

train_x = train_x.astype(np.float32) / 127.5 - 1
test_x = test_x.astype(np.float32) / 127.5 - 1

# train_x[:] = 0
# test_x[:] = 0
# train_x[:] = 1
# test_x[:] = 1

input_shape = nengo.transforms.ChannelShape(
    test_x[0].shape, channels_last=channels_last
)
assert input_shape.n_channels in (1, 3)
assert test_x[0].shape == input_shape.shape

test_x_flat = test_x.reshape((test_x.shape[0], 1, -1))

# --- create Nengo network
max_rate = 150
amp = 1.0 / max_rate
# rate_reg = 1e-2
rate_reg = 1e-3
rate_target = max_rate * amp  # must be in amplitude scaled units

relu = nengo.SpikingRectifiedLinear(amplitude=amp)
chip_neuron = nengo_loihi.neurons.LoihiLIF(amplitude=amp)

layer_confs = [
    dict(filters=4, kernel_size=1, strides=1, neuron=relu, on_chip=False),

    # dict(filters=4, kernel_size=3, strides=2, neuron=chip_neuron, block=(4, 4, 4)),

    # dict(filters=4, kernel_size=3, strides=1, neuron=chip_neuron, block=(8, 8, 4)),

    # --- these run 32x32
    # dict(filters=4, kernel_size=3, strides=2, neuron=chip_neuron, block=(15, 15, 4)),
    # dict(filters=8, kernel_size=3, strides=2, neuron=chip_neuron, block=(15, 15, 4)),

    # --- these hang 32x32
    dict(filters=4, kernel_size=3, strides=2, neuron=chip_neuron, block=(8, 8, 4)),
]

presentation_time = 0.2
present_images = nengo.processes.PresentInput(test_x_flat, presentation_time)

with nengo.Network(seed=0) as net:
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None

    nengo_loihi.add_params(net)  # allow setting on_chip

    # the input node that will be used to feed in input images
    inp = nengo.Node(present_images, label="input_node")

    connections = []
    transforms = []
    layer_probes = []
    shape_in = input_shape
    x = inp
    for k, layer_conf in enumerate(layer_confs):
        neuron_type = layer_conf.pop("neuron")
        on_chip = layer_conf.pop("on_chip", True)
        block = layer_conf.pop("block", None)
        name = layer_conf.pop("name", "layer%d" % k)

        # --- create layer transform
        # convolutional layer
        n_filters = layer_conf.pop("filters")
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
            channels_last=channels_last,
            init=nengo_dl.dists.Glorot(scale=1.0 / np.prod(kernel_size)),
            # init=nengo.dists.Uniform(1., 1.),
        )
        shape_out = transform.output_shape

        n_weights = np.prod(transform.kernel_shape)
        print(
            "%s: conv %s, stride %s, output %s (%d weights)"
            % (name, kernel_size, strides, shape_out.shape, n_weights)
        )

        # --- create layer output (Ensemble or Node)
        assert on_chip or block is None, "`block` must be None if off-chip"

        if neuron_type is None:
            assert not on_chip, "Nodes can only be run off-chip"
            y = nengo.Node(size_in=shape_out.size, label=name)
        else:
            ens = nengo.Ensemble(shape_out.size, 1, neuron_type=neuron_type, label=name)
            net.config[ens].on_chip = on_chip
            y = ens.neurons

            if block is not None:
                net.config[ens].block_shape = nengo_loihi.BlockShape(
                    block, shape_out.shape,
                )

        conn = nengo.Connection(x, y, transform=transform)
        net.config[conn].pop_type = 16

        transforms.append(transform)
        connections.append(conn)
        x = y
        shape_in = shape_out

    output_p = nengo.Probe(x[:10], synapse=None, label="output_p")

# n_presentations = 10
# n_presentations = 6
# n_presentations = 2
n_presentations = len(test_x_flat)

for conn in net.all_connections:
    conn.synapse = None

sim_time = n_presentations * presentation_time

with nengo_loihi.Simulator(net) as sim:
    for block in sim.model.blocks:
        n_output_axons = sum(a.axon_slots() for a in block.axons)
        n_input_axons = sum(s.n_axons for s in block.synapses)
        synapse_bits = sum(s.bits() for s in block.synapses)
        print("Block %s: %d compartments, %d input axons, %d output axons, %d bits synapse mem"
              % (block, block.compartment.n_compartments, n_input_axons, n_output_axons, synapse_bits)
        )

    print("%d blocks" % len(sim.model.blocks))

    print("Running for %0.3f sim time" % sim_time)
    sim.run(sim_time)

print(sim.data[output_p][-1])
