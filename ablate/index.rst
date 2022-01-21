****************
Ablating neurons
****************

These examples emulate the ablation of neurons
by setting the encoders of those neurons to 0,
and setting the bias of those neurons to a large negative number.

The function of interest is::

    def ablate_ensemble(ens, proportion, sim, bias=True):
        """Ablate a proportion of the neurons in an ensemble.

        The ablation is done by setting the encoder and gain associated
        with a neuron to zero. Since no input current being injected,
        the neuron will generally be silent. However, if there is direct
        current injected with a neuron-to-neuron connection, then the
        cell may still fire. To counter that in most cases, we set the
        bias associated with the neuron to a large negative value.
        """

        n_neurons = min(int(ens.n_neurons * proportion), ens.n_neurons)
        idx = np.random.choice(np.arange(ens.n_neurons), replace=False, size=n_neurons)

        encoder_sig = sim.signals[sim.model.sig[ens]['encoders']]
        encoder_sig.setflags(write=True)
        encoder_sig[idx] = 0.0
        encoder_sig.setflags(write=False)

        if bias:
            bias_sig = sim.signals[sim.model.sig[ens.neurons]['bias']]
            bias_sig.setflags(write=True)
            bias_sig[idx] = -1000

Examples
========

The follow examples show how this can be applied
in a simple model and a more complicated SPA model.

.. toctree::
   :maxdepth: 1

   ablate-ensemble
   ablate-spa

Requirements
============

- `jupyter <http://jupyter.readthedocs.io/en/latest/install.html>`_
- `matplotlib <https://matplotlib.org/users/installing.html>`_
- `nengo>=3.0 <https://www.nengo.ai/getting-started/>`_

License
=======

Copyright (c) 2017-2022 Applied Brain Research

.. include:: ../NENGO_LICENSE.rst
