import pickle
import nengo
import numpy as np

import audio_stream
from utils import allowed_text, id_to_char

# set this to use your chosen audio input device
if not audio_stream.started():
    audio_stream.start('Plantronics Blackwire 325.1')

n_inputs = 390
n_outputs = 29
n_neurons = 256

audio_stream.min_thresh = 200
audio_stream.voc_thresh = 800


def default_value(t, x):
    '''Node function to predict blank character by default'''
    y = x + 100  # just for visualizing the filtered output value
    y[-1] += 0.5  # boost the baseline for blank char
    return y


with open('demo_params.pkl', 'rb') as pfile:
    params = pickle.load(pfile)

# core speech model for keyword spotting
with nengo.Network() as model:
    model.config[nengo.Connection].synapse = None

    neuron_type = nengo.LIF(
        tau_rc=0.02, tau_ref=0.001, amplitude=0.002)

    audio_input = audio_stream.AudioInput()
    audio_rms = nengo.Node(lambda t: audio_stream.rmses, label='Audio RMS')

    # below is the core model architecture
    inp = nengo.Node(None, size_in=n_inputs)
    out = nengo.Node(None, size_in=n_outputs)
    bias = nengo.Node(1)

    nengo.Connection(audio_input, inp, synapse=None)

    layer_1 = nengo.Ensemble(n_neurons=n_neurons, dimensions=1,
                             neuron_type=neuron_type,
                             gain=params['x_c_0']['gain'],
                             bias=params['x_c_0']['bias'],
                             label='Layer 1')

    layer_2 = nengo.Ensemble(n_neurons=n_neurons, dimensions=1,
                             neuron_type=neuron_type,
                             gain=params['x_c_1']['gain'],
                             bias=params['x_c_1']['bias'],
                             label='Layer 2')

    nengo.Connection(
        inp, layer_1.neurons, transform=params['input_node -> x_c_0'])

    nengo.Connection(
        layer_1.neurons, layer_2.neurons, transform=params['x_c_0 -> x_c_1'])

    nengo.Connection(
        layer_2.neurons, out, transform=params['x_c_1 -> char_output'])

    nengo.Connection(
        bias, out,
        transform=np.expand_dims(params['char_output_bias'], axis=1))

    # filter and bias towards predicting blank symbol by default
    filtered_output = nengo.Node(
        default_value, size_in=29, size_out=29, label='Filtered Output')
    nengo.Connection(out, filtered_output, synapse=0.005)


# Slightly modified version of Terry's code for HTML visualization in GUI
with model:
    winsize = 10
    blank_count = [0]
    display_count = [0]
    window_index = [0]
    window = np.zeros(winsize, dtype=int)

    def result_func(t, x):
        '''Node function for displaying predicted text sequence'''
        window[window_index[0]] = np.argmax(x)
        window_index[0] = (window_index[0]+1) % winsize

        char = id_to_char[np.argmax(np.bincount(window))]

        if blank_count[0] > 1000:
            result_func._nengo_html_ = 'Prediction: \n'
            result_func._nengo_html_ += char

        if char != '-':
            if (len(result_func._nengo_html_) == 0 or
                    result_func._nengo_html_[-1] != char):
                result_func._nengo_html_ += char

            blank_count[0] = 0
        else:
            blank_count[0] += 1

        return None

    def decision_func(t, x):
        '''Node function for displaying acceptance decision'''
        decision_func._nengo_html_ = 'Decision: '
        phrase = result_func._nengo_html_.split()[-1]

        if phrase in allowed_text and display_count[0] < 1000:
            decision_func._nengo_html_ += (
                '\n <font color="green"> Accepted!</font>')
            display_count[0] += 1
        else:
            decision_func._nengo_html_ += (
                '\n <font color="red"> Not accepted!</font>')
            display_count[0] = 0

        return None

    result_func._nengo_html_ = 'Prediction: <hr/>\n'
    decision_func._nengo_html_ = ''

    result = nengo.Node(result_func, size_in=29, label='Characters')
    nengo.Connection(filtered_output, result, synapse=None)

    accept = nengo.Node(decision_func, size_in=29, label="Decision")
    nengo.Connection(filtered_output, accept, synapse=None)
