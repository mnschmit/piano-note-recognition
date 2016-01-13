#!/usr/bin/python

from sys import argv

filename = argv[1]
midi_filename = "/home/martin/uni/music/miniproject/reference/ty_januarMINp_align.mid"
if len(argv) > 2:
    midi_filename = argv[2]

n_components=None
if len(argv) > 3:
    n_components = int(argv[3])

from librosa import load, cqt, logamplitude
import numpy as np

# load an audio file (with samplerate)
x, sr = load(filename)

# compute constant-Q transform (~ pitch-based STFT)
hop_size = 512
C = cqt(x, sr=sr, hop_length=hop_size)
#C = stft(x, hop_length=hop_size)

# try some midi visualization
from Midi import midi_matrix

midi_mat = midi_matrix(midi_filename)

# NMF

#V = np.log10(1 + 100000 * C**2)
V = np.abs(C)

from librosa.decompose import decompose

comps, acts = decompose(V, n_components=n_components, sort=True)

# visualisation matters
import matplotlib.pyplot as plt
from librosa.display import specshow
import matplotlib.gridspec as gridspec

plt.close('all')

plt.subplot2grid((4, 2), (0,0), colspan=2)
specshow(midi_mat, sr=sr, x_axis='time', y_axis='cqt_note')
plt.title('midi visualization')

plt.subplot2grid((4, 2), (1,0), colspan=2)
specshow(V, sr=sr, x_axis='time', y_axis='cqt_note')
#plt.ylabel('pitches')
plt.colorbar(format='%+2.0f dB')
plt.title('Input Constant-Q power spectrum')

plt.subplot2grid((4, 2), (2,0))
specshow(comps, y_axis='cqt_note')
#plt.ylabel('pitches')
plt.xlabel('Index')
plt.title('Learned Components')

plt.subplot2grid((4, 2), (2,1))
specshow(acts, x_axis='time')
plt.ylabel('Components')
plt.title('Activations')

plt.subplot2grid((4, 2), (3,0), colspan=2)
V_approx = comps.dot(acts)
specshow(V_approx, sr=sr, x_axis='time', y_axis='cqt_note')
#plt.ylabel('pitches')
plt.colorbar(format='%+2.0f dB')
plt.title('Reconstructed spectrum')

plt.tight_layout()

plt.show()

