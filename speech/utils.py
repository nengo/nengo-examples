"""Includes code to do Mel-frequency cepstral analysis.
Most of this is based on the ``python-speech-features`` library written by
James Lyons and available at
https://github.com/jameslyons/python_speech_features.
The code is MIT licensed, so we can include it here
(https://github.com/jameslyons/python_speech_features/blob/master/LICENSE)
"""
import string
import math
import os
import shutil
import zipfile
import requests

import numpy as np
import tensorflow as tf

from nengo.utils.compat import range
from scipy.fftpack import dct

allowed_text = ["loha", "alha", "aloa", "aloh", "aoha", "aloha"]
id_to_char = np.array([x for x in string.ascii_lowercase + "\" -|"])


def ce_loss(x, y, weight=1):
    '''Cross entropy loss function for training keyword spotter'''
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=x, labels=y)
    return weight * tf.reduce_sum(loss)


def weight_init(shape):
    '''Convenience function for randomly initializing weights'''
    weights = np.random.uniform(-0.05, 0.05, size=shape)
    return weights


def merge(chars):
    '''Merge repeated characters and strip blank CTC symbol'''
    acc = ["-"]
    for c in chars:
        if c != acc[-1]:
            acc.append(c)

    acc = [c for c in acc if c != "-"]
    return "".join(acc)


def predict_text(sim, probe, n_steps, p_time=10):
    '''Predict a text transcription from the current simulation state'''
    n_frames = int(n_steps / p_time)
    char_data = sim.data[probe]
    n_chars = char_data.shape[1]

    # reshape to seperate out each window frame that was presented
    char_out = np.reshape(char_data, (n_frames, p_time, n_chars))

    # take most ofter predicted char over each frame presentation interval
    char_ids = np.argmax(char_out, axis=2)
    char_ids = [np.argmax(np.bincount(i)) for i in char_ids]

    text = merge(''.join([id_to_char[i] for i in char_ids]))
    text = merge(text)  # merge repeats to help autocorrect

    return text


def create_stream(stream, dt=0.001):
    '''Create a streaming function for sending data into Nengo network'''
    def play_stream(t, stream=stream):

        ti = int(t / dt)
        return stream[ti % len(stream)]

    return play_stream


def download_and_unzip(gdrive_id, filename, path):
    '''Function from SO for dealing with GDrive downloads'''
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keepalive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': gdrive_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': gdrive_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, filename)

    with zipfile.ZipFile(filename) as zfile:
        for item in zfile.namelist():
            filename = os.path.basename(item)
            # skip directories
            if not filename:
                continue

            source = zfile.open(item)
            target = open(os.path.join(path, filename), 'wb')
            with source, target:
                shutil.copyfileobj(source, target)


def mfcc(audio, fs=16000, window_dt=0.025, dt=0.01, n_cepstra=13,
         n_filters=26, n_fft=512, minfreq=0, maxfreq=None, preemph=0.97,
         lift=22, energy=True, n_derivatives=0, deriv_spread=2):
    """Compute MFCC features from an audio signal.

    Parameters
    ----------
    audio : array_like (N, 1)
        The audio signal from which to compute features.
    fs : float, optional
        The samplerate of the signal we are working with. Default: 16000
    window_dt : float, optional
        The length of the analysis window in seconds.
        Default: 0.025 (25 milliseconds)
    dt : float, optional
        The step between successive windows in seconds.
        Default: 0.01 (10 milliseconds)
    n_cepstra : int, optional
        The number of cepstral coefficients to return. Default: 13
    n_filters : int, optional
        The number of filters in the filterbank. Default: 26
    n_fft : int, optional
        The FFT size. Default: 512
    minfreq : int, optional
        Lowest band edge of Mel filters, in Hz. Default: 0
    maxfreq : int, optional
        highest band edge of mel filters, in Hz. Default: fs / 2
    preemph : float, optional
        Apply preemphasis filter with preemph as coefficient; 0 is no filter.
        Default: 0.97
    lifter : float, optional
        Apply a lifter to final cepstral coefficients; 0 is no lifter.
        Default: 22.
    energy : bool, optional
        If this is true, the zeroth cepstral coefficient is replaced with the
        log of the total frame energy. Default: True
    n_derivatives : int, optional
        The number of derivatives to include in the feature vector.
        Affects the shape of the returned array. Default: 0
    deriv_spread : int, optional
        The spread of the derivatives to includ in the feature vector.
        Greater spread uses more frames to compute the derivative.
        Default: 2

    Returns
    -------
    A numpy array of shape (audio.shape[0], n_cepstra * (1 + n_derviatives)
    containing features. Each row holds 1 feature vector.
    """
    feat, energy_ = fbank(
        audio, fs, window_dt, dt, n_filters, n_fft, minfreq, maxfreq, preemph)
    feat = np.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :n_cepstra]
    feat = lifter(feat, lift)
    if energy:
        # replace first cepstral coefficient with log of frame energy
        feat[:, 0] = np.log(energy_)

    target = feat
    derivs = []
    for i in range(n_derivatives):
        derivs.append(derivative(target, deriv_spread))
        target = derivs[-1]
    return np.hstack([feat] + derivs)


def fbank(audio, fs=16000, window_dt=0.025, dt=0.01,
          n_filters=26, n_fft=512, minfreq=0, maxfreq=None, preemph=0.97):
    """Compute Mel-filterbank energy features from an audio signal.

    Returns
    -------
    features : array_like (audio.shape[0], n_filters)
        Features; each row holds 1 feature vector.
    energy : array_like
        Total energy in each frame, unwindowed.
    """
    maxfreq = fs // 2 if maxfreq is None else maxfreq
    audio = preemphasis(audio, preemph)
    frames = framesig(audio, window_dt*fs, dt*fs)
    pspec = powspec(frames, n_fft)
    energy = np.sum(pspec, axis=1)  # stores the total energy in each frame
    # if energy is zero, we get problems with log
    energy[energy == 0] = np.finfo(float).eps

    fb = get_filterbanks(n_filters, n_fft, fs, minfreq, maxfreq)
    feat = np.dot(pspec, fb.T)  # compute the filterbank energies
    # if feat is zero, we get problems with log
    feat[feat == 0] = np.finfo(float).eps

    return feat, energy


def logfbank(audio, fs=16000, window_dt=0.025, dt=0.01,
             n_filters=26, n_fft=512, minfreq=0, maxfreq=None, preemph=0.97):
    """Compute log Mel-filterbank energy features from an audio signal."""
    feat, energy = fbank(
        audio, fs, window_dt, dt, n_filters, n_fft, minfreq, maxfreq, preemph)
    return np.log(feat)


def ssc(audio, fs=16000, window_dt=0.025, dt=0.01,
        n_filters=26, n_fft=512, minfreq=0, maxfreq=None, preemph=0.97):
    """Compute Spectral Subband Centroid features from an audio signal."""
    maxfreq = fs // 2 if maxfreq is None else maxfreq
    audio = preemphasis(audio, preemph)
    frames = framesig(audio, window_dt*fs, dt*fs)
    pspec = powspec(frames, n_fft)
    # if things are all zeros we get problems
    pspec[pspec == 0] = np.finfo(float).eps

    fb = get_filterbanks(n_filters, n_fft, fs, minfreq, maxfreq)
    feat = np.dot(pspec, fb.T)  # compute the filterbank energies
    R = np.tile(np.linspace(1, fs // 2, pspec.shape[1]),
                (pspec.shape[0], 1))
    return np.dot(pspec * R, fb.T) / feat


def get_filterbanks(n_filters=20, n_fft=512, fs=16000,
                    minfreq=0, maxfreq=None):
    """Compute a Mel-filterbank.

    The filters are stored in the rows, the columns correspond to fft bins.
    The filters are returned as an array of shape (n_filters, n_fft/2 + 1).

    Parameters
    ----------
    n_filters : int, optional
        The number of filters in the filterbank. Default: 20
    n_fft : int, optional
        The FFT size. Default: 512
    fs : float, optional
        The samplerate of the signal we are working with. Affects mel spacing.
        Default: 16000
    minfreq : int, optional
        Lowest band edge of mel filters, in Hz. Default: 0
    maxfreq : int, optional
        highest band edge of mel filters, in Hz. Default: fs / 2

    Returns
    -------
    Numpy array of shape (n_filters, n_fft/2 + 1) containing the filterbank.
    Each row holds 1 filter.
    """
    maxfreq = fs // 2 if maxfreq is None else maxfreq
    assert maxfreq <= fs // 2, "maxfreq is greater than fs/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(minfreq)
    highmel = hz2mel(maxfreq)
    melpoints = np.linspace(lowmel, highmel, n_filters + 2)
    # Our points are in Hz, but we use fft bins, so we have to convert
    # from Hz to fft bin number
    fbin = np.floor((n_fft + 1) * mel2hz(melpoints) / fs)

    fbank = np.zeros([n_filters, int(n_fft/2+1)])
    for j in range(n_filters):
        for i in range(int(fbin[j]), int(fbin[j + 1])):
            fbank[j, i] = (i - fbin[j]) / (fbin[j + 1] - fbin[j])
        for i in range(int(fbin[j + 1]), int(fbin[j + 2])):
            fbank[j, i] = (fbin[j + 2] - i) / (fbin[j + 2] - fbin[j + 1])
    return fbank


def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra.

    Liftering increases the magnitude of the high frequency DCT coefficients.

    Parameters
    ----------
    cepstra : array_like (numframes, n_cepstra)
        The matrix of Mel-cepstra.
    L : float, optional
        The liftering coefficient to use; L <= 0 disables the lifter.
        Default: 22
    """
    if L <= 0:
        # values of L <= 0 do nothing
        return cepstra
    nframes, ncoeff = cepstra.shape
    n = np.arange(ncoeff)
    lift = 1 + (L / 2) * np.sin(np.pi * n / L)
    return lift * cepstra


def framesig(sig, frame_len, frame_step, winfunc=lambda x: np.ones(x)):
    """Frame a signal into overlapping frames.

    Parameters
    ----------
    sig : array_like
        The audio signal to frame.
    frame_len : int
        Length of each frame, in samples.
    frame_step : int
        Number of samples after the start of the previous frame
        that the next frame should begin.
    winfunc : function, optional
        The analysis window to apply to each frame. Default: No window

    Returns
    -------
    Numpy array of frames with shape (NUMFRAMES, frame_len).
    """

    slen = len(sig)
    frame_len = int(round(frame_len))
    frame_step = int(round(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes - 1) * frame_step + frame_len)
    padsignal = np.concatenate((sig, np.zeros(padlen - slen)))

    indices = (np.tile(np.arange(frame_len), (numframes, 1)) +
               np.tile(np.arange(0, numframes * frame_step, frame_step),
               (frame_len, 1)).T.astype(dtype=np.int32))

    frames = padsignal[indices]
    win = np.tile(winfunc(frame_len), (numframes, 1))
    return frames * win


def magspec(frames, N_FFT):
    """Compute the magnitude spectrum of each frame in frames.

    Parameters
    ----------
    frames : array_like
        The array of frames. Each row is a frame.
    N_FFT : int
        The FFT length to use. If N_FFT > frame_len, frames are zero-padded.

    Returns
    -------
    If frames has shape (N, D), output will be shape (N, N_FFT).
    Each row will be the magnitude spectrum of the corresponding frame.
    """
    complex_spec = np.fft.rfft(frames, N_FFT)
    return np.absolute(complex_spec)


def powspec(frames, N_FFT):
    """Compute the power spectrum of each frame in frames.

    Parameters
    ----------
    frames : array_like
        The array of frames. Each row is a frame.
    N_FFT : int
        The FFT length to use. If N_FFT > frame_len, frames are zero-padded.

    Returns
    -------
    If frames has shape (N, D), output will be shape (N, N_FFT).
    Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0 / N_FFT * np.square(magspec(frames, N_FFT))


def logpowspec(frames, N_FFT, norm=True):
    """Compute the log power spectrum of each frame in frames.

    Parameters
    ----------
    frames : array_like
        The array of frames. Each row is a frame.
    N_FFT : int
        The FFT length to use. If N_FFT > frame_len, frames are zero-padded.
    norm : bool, optional
        If True, the log power spectrum is normalised so that the max value
        (across all frames) is 1. Default: True

    Returns
    -------
    If frames has shape (N, D), output will be shape (N, N_FFT).
    Each row will be the log power spectrum of the corresponding frame.
    """
    ps = powspec(frames, N_FFT)
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * np.log10(ps)
    return lps - np.max(lps) if norm else lps


def preemphasis(signal, coeff=0.95):
    """Preemphasize the input signal.

    Parameters
    ----------
    signal : array_like
        The signal to filter.
    coeff : float, optional
        The preemphasis coefficient; 0 for no filtering. Default: 0.95

    Returns
    -------
    The filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def hz2mel(hz):
    """Convert a value in Hertz to Mels."""
    return 2595. * np.log10(1 + hz / 700.0)


def mel2hz(mel):
    """Convert a value in Mels to Hertz."""
    return 700. * (10. ** (mel / 2595.0) - 1)


def hz2st(hz, reference=16.35159783):
    """Convert hertz to semi-tones, relative to musical note C0."""
    if hz < 1.0:
        return 1.0
    return 12 * np.log2(hz / reference)


def st2hz(st, reference=16.35159783):
    """Convert semi-tones to hertz, relative to musical note C0."""
    return reference * np.power(2, st / 12.)


def derivative(feat, spread):
    assert feat.ndim == 2
    if feat.shape[0] == 1:
        # Can't do derivative of one sample
        return feat
    spread = min(spread, feat.shape[0] - 1)
    out = np.zeros_like(feat)
    for i in range(1, spread + 1):
        plus = np.roll(feat, -i, axis=0)
        plus[-i:] = plus[-i-1]
        minus = np.roll(feat, i, axis=0)
        minus[:i] = 0.
        out += plus - minus
    return out / (2 * np.sum(np.arange(1, spread + 1)))


def rescale(val, old_min, old_max, new_min, new_max):
    old_range = old_max - old_min
    new_range = new_max - new_min
    return (((val - old_min) * new_range) / old_range) + new_min
