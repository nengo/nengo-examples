***************************
Speech Recognition in Nengo
***************************

This repository contains two examples illustrating speech recognition in spiking
neural networks. In both cases, the neural network performs a keyword spotting
task, discriminating a target phrase ("aloha") from other kinds of speech.

The first example takes a pretrained keyword spotting network and runs it with
real-time audio input collected via a microphone or headset. The second example
is a Jupyter notebook illustrating how to train a keyword spotter from scratch
using Nengo DL, before deploying this network on the Loihi neuromorphic chip.


**Installation**
~~~~~~~~~~~~~~~~

To start, we recommend that you make an Anaconda environment and follow the instructions below. `Anaconda <https://www.anaconda.com/download/>`_, greatly simplifies the installation of some of the audio preprocessing tools you will need to run the example involving live input from a microphone.

.. code:: shell

    # create and activate a new python environment 
    conda create -n speech-examples python=3.6
    source activate speech-examples

    # install Nengo packages 
    pip install nengo
    pip install nengo-dl
    pip install nengo-loihi
    pip install nengo-gui

    # install packages for audio feature processing
    pip install git+http://people.csail.mit.edu/hubert/git/pyaudio.git#egg=pyaudio
    pip install requests
    conda install -c conda-forge librosa
    conda install scipy

    # install Jupyter Notebook
    pip install jupyter 

    # open a notebook for the first example 
    jupyter notebook

    # or open the GUI for the second example
    nengo live_demo_model.py

To run the GUI demo with live audio input, you'll need to specify the audio device's name at the top of the script visible in the GUI based on the options presented in the bottom right panel of the GUI window (you may need to restart the GUI aftewards). You may also need to adjust the variables :code:`min_thresh` and :code:`voc_thresh` to better suit the voice activity detector to the specifics of your hardware setup. Best performance is likely to be found with a headset microphone.

