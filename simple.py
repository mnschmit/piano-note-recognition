#!/usr/bin/python

from sys import argv

filename = argv[1]

n_components=2
if len(argv) > 2:
    n_components = int(argv[2])

from librosa import load, cqt, logamplitude
import numpy as np

# load an audio file (with samplerate)
x, sr = load(filename)

# compute constant-Q transform (~ pitch-based STFT)
C = cqt(x, sr=sr)

# NMF

V = np.log10(1 + 100000 * C**2)

from librosa.decompose import decompose

comps, acts = decompose(V, n_components=n_components, sort=True)

# visualisation matters
import matplotlib.pyplot as plt
from librosa.display import specshow

plt.subplot(3, 1, 1)
specshow(V, sr=sr, x_axis='time', y_axis='cqt_note')
#plt.ylabel('pitches')
plt.colorbar(format='%+2.0f dB')
plt.title('Input Constant-Q power spectrum')

plt.subplot(3, 2, 3)
specshow(comps, y_axis='cqt_note')
#plt.ylabel('pitches')
plt.xlabel('Index')
plt.title('Learned Components')

plt.subplot(3, 2, 4)
specshow(acts, x_axis='time')
plt.ylabel('Components')
plt.title('Activations')

plt.subplot(3, 1, 3)
V_approx = comps.dot(acts)
specshow(V_approx, sr=sr, x_axis='time', y_axis='cqt_note')
#plt.ylabel('pitches')
plt.colorbar(format='%+2.0f dB')
plt.title('Reconstructed spectrum')

plt.tight_layout()

plt.show()

