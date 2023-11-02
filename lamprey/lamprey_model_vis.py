import nengo
import numpy as np
import scipy.integrate as sp


def phi(z, m):
    return np.exp(-np.square(z - (m / 10.0)) / np.square(0.5))


def Phi(z):
    return [
        1.0,
        np.sin(2.0 * np.pi * z),
        np.cos(2.0 * np.pi * z),
        np.sin(4.0 * np.pi * z),
    ]


def get_coefficients(n, m):
    f = lambda x: Phi(x)[n] * phi(x, m)
    return sp.quad(f, 0, 1)[0]


Gamma = np.zeros((3, 10))
for m in range(10):
    for n in range(3):
        Gamma[n, m] = get_coefficients(n, m)
    ### end for
### end for
Gamma_inv = np.linalg.pinv(Gamma)

Z_mat = np.zeros((10, 10))
for z_idx, z in enumerate(np.linspace(0, 1, 10)):
    for m in range(10):
        Z_mat[z_idx, m] = phi(z, m)
    ### end for
### end for


def T(x):
    return np.dot(Z_mat, np.dot(Gamma_inv, x))


class Lamprey:
    def __init__(self):
        self.state = np.zeros(9)
        self._nengo_html_ = ""
        self.dt = 0.001

    def __call__(self, t, x):
        self.state = +np.diff(x)
        y1 = 5 * self.state + 50
        x1 = np.linspace(10, 90, 10)
        self._nengo_html_ = """
            <svg width="100%" height="100%" viewbox="0 0 100 100">
                <circle cx="{x1[1]}" cy="{y1[1]}" r="1" stroke="black"/>
                <circle cx="{x1[2]}" cy="{y1[2]}" r="1" stroke="black"/>
                <circle cx="{x1[3]}" cy="{y1[3]}" r="1" stroke="black"/>
                <circle cx="{x1[4]}" cy="{y1[4]}" r="1" stroke="black"/>
                <circle cx="{x1[5]}" cy="{y1[5]}" r="1" stroke="black"/>
                <circle cx="{x1[6]}" cy="{y1[6]}" r="1" stroke="black"/>
                <circle cx="{x1[7]}" cy="{y1[7]}" r="1" stroke="black"/>
                <circle cx="{x1[8]}" cy="{y1[8]}" r="1" stroke="black"/>
                <polyline stroke="black" fill="none"
                points="{x1[0]},{y1[0]}, {x1[1]},{y1[1]}, {x1[2]},{y1[2]},
                {x1[3]},{y1[3]}, {x1[4]},{y1[4]}, {x1[5]},{y1[5]},
                {x1[6]},{y1[6]}, {x1[7]},{y1[7]}, {x1[8]},{y1[8]}"/>
                <circle cx="{x1[0]}" cy="{y1[0]}" r="3" stroke="black" fill="red"/>
            </svg>
        """.format(
            **locals()
        )


lamp = Lamprey()

with nengo.Network("basic_lamprey", seed=1) as model:
    gauss_dist = nengo.dists.Gaussian(mean=0, std=0.1)
    white_noise = nengo.processes.WhiteNoise(dist=gauss_dist)

    tau = 0.2
    damp0 = 0.1
    freq_hz = 3.0
    freq = freq_hz * 2 * np.pi

    a = nengo.Ensemble(
        n_neurons=500, dimensions=3, radius=1, noise=white_noise, label="cpg"
    )

    M_d = np.array([[-damp0, freq, -damp0], [-freq, 0, 0], [-damp0, -freq, -damp0]])
    M_i = np.array([[0.5, 0, -0.5], [0, 1, 0], [-0.5, 0, 0.5]])

    def feedback_func(x):
        return tau * np.dot(M_d, x) + 1.05 * x

    nengo.Connection(a, a, function=feedback_func, synapse=tau)

    def stim_func(t):
        return np.array([1.0, 1.0, 1.0]) if t < 1.0 else np.array([0, 0, 0])

    ### end stim_func
    u = nengo.Node(stim_func)

    def kick_func(x):
        return tau * np.dot(M_i, x)

    nengo.Connection(u, a, function=kick_func, synapse=tau)

    tensions = nengo.Ensemble(n_neurons=100, dimensions=10, neuron_type=nengo.Direct())
    nengo.Connection(a, tensions, function=T)

    lamprey = nengo.Node(lamp, size_in=10)
    nengo.Connection(tensions, lamprey)

    ## Log data
### end with
