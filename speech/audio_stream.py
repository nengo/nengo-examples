import audioop
import wave
import uuid

import nengo
import pyaudio
import librosa
import numpy as np

from utils import mfcc

# configure the streaming with pyaudio
channels = 1
rate = 16000
n_frames = 15
dtype = pyaudio.paInt16

audio_driver = pyaudio.PyAudio()
driver_info = audio_driver.get_host_api_info_by_index(0)
numdevices = driver_info.get('deviceCount')

# configure callback for capturing new audio data
frame_count = 1024
zero_samples = np.zeros(frame_count)
sleep_start = 0
min_thresh = 200
voc_thresh = 800

# global variables for audio features, RMSE vals, stream status
stream = None
queue = []
rmses = [0, 0, 0]


class FrameBuffer:
    '''A sliding buffer that stores audio input from stream'''
    def __init__(self, size):
        self.size = size
        self.data = []

    def extend(self, x):
        '''Add input to the buffer and delete old stuff if size maxed out'''
        self.data.append(x)
        if len(self.data) > self.size:
            self.data = self.data[-self.size:]


def get_device_list():
    '''List all devices for which audio input can be collected'''
    all_devices = []
    for i in range(numdevices):
        dev_idx = audio_driver.get_device_info_by_host_api_device_index(0, i)
        all_devices.append(dev_idx.get('name'))

    return all_devices


def get_device_index_by_name(name):
    '''Return the index of the device targeted by GUI script for audio input'''
    for idx, device_name in enumerate(get_device_list()):
        if device_name.startswith(name):
            return idx

    print(get_device_list())
    raise Exception('Unknown device: %s' % name)


buff = FrameBuffer(size=16)  # size sets callbacks to keep, each 1024 samples


def callback(in_data, frame_count, time_info, status):
    '''Function called whenever new data is available from audio device'''
    global queue
    global buff
    global sleep_start
    global zero_seamples

    samples = zero_samples
    # add current input to window buffer
    buff.extend(in_data)

    if len(buff.data) > 6:
        onset_rms = audioop.rms(b''.join(buff.data[:1]), 2)
        midpoint_rms = audioop.rms(b''.join(buff.data[1:-1]), 2)
        offset_rms = audioop.rms(b''.join(buff.data[-1:]), 2)

        # set this for plotting in Nengo GUI
        rmses[0] = onset_rms
        rmses[1] = midpoint_rms
        rmses[2] = offset_rms

        # condition implements a naive voice activity detector
        if (onset_rms < min_thresh and offset_rms < min_thresh and
                midpoint_rms > voc_thresh):
            # once detected, sleep for 800ms to avoid repeating phrase
            if time_info['current_time'] - sleep_start > 0.8:
                sleep_start = time_info['current_time']
                idx = str(uuid.uuid4())
                # mild hack to get numpy of waveform, save to wav then load
                print('Phrase detected!! Adding data to queue...')
                waveFile = wave.open('audio.wav', 'wb')
                waveFile.setnchannels(channels)
                waveFile.setsampwidth(audio_driver.get_sample_size(dtype))
                waveFile.setframerate(rate)
                waveFile.writeframes(b''.join(buff.data))
                waveFile.close()

                # load and trim waveform
                audio, _ = librosa.load('audio.wav', sr=16000)
                trimmed_audio, _ = librosa.effects.trim(audio, top_db=35)

                # get mfccs and convert to feature stream for chip/nengo DL
                mfccs = mfcc(trimmed_audio, n_cepstra=26)
                n_windows = mfccs.shape[0] - n_frames
                all_windows = []

                for idx in range(n_windows):
                    feature_window = mfccs[idx:idx+n_frames, :].flatten()
                    all_windows.append(feature_window)

                queue.append(np.vstack(all_windows))

    return (samples, pyaudio.paContinue)


class AudioInput(nengo.Node):
    '''Subclass of Nengo Node for streaming in captured buffer data'''
    def __init__(self):
        self.item = None
        self.index = 0
        self.delay_length = 300
        self.delay = self.delay_length
        self.steps = 0
        self.count = 0
        super(AudioInput, self).__init__(
            self.update, size_in=0, size_out=390, label="Audio Input")

    def update(self, t):
        '''Function called at every time step of Nengo simulation'''
        # return zeros if delay period is active, then pop from queue
        if self.delay > 0:
            self.delay -= 1
            if self.delay == 0:
                if len(queue) > 0:
                    self.item = queue.pop(0)
                    self.index = 0
                    self.count += 1

            return np.zeros(390)

        # return zeros if feature queue is empty
        elif self.item is None:
            self.delay = 1

            return np.zeros(390)
        # present each feature vector for 10 timesteps, stepping through MFCC
        else:
            features = self.item[self.index]
            self.steps += 1
            if self.steps == 10:
                self.steps = 0
                self.index += 1
                if self.index >= len(self.item):
                    self.item = None
                    self.delay = self.delay_length

            return features


def start(device_name):
    '''Start the audio stream callback'''
    global stream
    device_index = get_device_index_by_name(device_name)
    print(device_index)

    stream = audio_driver.open(format=dtype,
                               channels=channels,
                               rate=rate,
                               input=True,
                               output=True,
                               frames_per_buffer=frame_count,
                               input_device_index=device_index,
                               output_device_index=None,
                               stream_callback=callback)

    stream.start_stream()


def started():
    '''Check if audio stream is set up'''
    return stream is not None


def stop():
    '''Stop the audio stream callback and shut down'''
    global stream
    if stream is not None:
        stream.stop_stream()
        stream.close()
        stream = None

    audio_driver.terminate()
