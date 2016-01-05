#!/usr/bin/python

from sys import argv

filename = argv[1]

from librosa import load, stft
import numpy as np

# load an audio file (with samplerate)
x, fs = load(filename)

window_length = 2048
# compute normal stft
X = stft(x, n_fft=window_length)

Y = np.abs(X) ** 2

# transform the frequencies into pitches

### first the frequencies, then the time! ###
K, N = X.shape

from Lab1 import computeP
# Y_LF
V = np.zeros((128, N))
for n in range(N):
    for p in range(128):
        V[p][n] = sum(map(lambda k: Y[k][n], computeP(fs, window_length, K, p)))

# NMF

from librosa.decompose import decompose

comps, acts = decompose(V, n_components=8, sort=True)

# visualisation matters
import matplotlib.pyplot as plt
from librosa.display import specshow
from librosa import logamplitude

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
specshow(V, x_axis='time')
plt.ylabel('pitches')
plt.colorbar(format='%+2.0f dB')
plt.title('Input power spectrogram')

plt.subplot(3, 2, 3)
specshow(comps)
plt.ylabel('pitches')
plt.xlabel('Index')
plt.title('Learned Components')

plt.subplot(3, 2, 4)
specshow(acts, x_axis='time')
plt.ylabel('Components')
plt.title('Activations')

plt.subplot(3, 1, 3)
V_approx = comps.dot(acts)
specshow(V_approx, x_axis='time')
plt.ylabel('pitches')
plt.colorbar(format='%+2.0f dB')
plt.title('Reconstructed spectrogram')

plt.tight_layout()

plt.show()

