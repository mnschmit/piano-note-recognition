#!/usr/bin/python

from sys import argv

filename = argv[1]

from librosa import load, stft

x, fs = load(filename)

X = stft(x, fs)

# visualisation matters
import matplotlib.pyplot as plt
from librosa.display import specshow

plt.figure(figsize=(12, 8))
specshow(X, fs)
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
plt.show()
