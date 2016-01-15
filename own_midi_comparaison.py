#!/usr/bin/python

usage='''
Usage: own_midi_comparaison.py filename.wav filename.mid [pitch_max pitch_min]

       Mandatory arguments : two files to compare
       Optional arguments  : pitch_max (biggest pitch considered), pitch_min (smalles pitch considered) 
'''

import sys
from librosa import load, stft, logamplitude, note_to_midi, midi_to_hz
import numpy as np


if len(sys.argv) <= 2:
    print usage
    sys.exit(-1)

filename = sys.argv[1]

midi_filename = sys.argv[2]

pitch_max = note_to_midi('C7')
if len(sys.argv) > 3:
    pitch_max = note_to_midi(sys.argv[3])

pitch_min = note_to_midi('C1')
if len(sys.argv) > 4:
    pitch_min = note_to_midi(sys.argv[4])

pitches = range(pitch_min, pitch_max + 1)
#pitches = note_to_midi(['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'])

# load an audio file (with samplerate)
x, sr = load(filename)

# compute normal STFT
n_components = len(pitches)
n_fft = 2048
hop_length = n_fft
X = stft(x, n_fft=n_fft, hop_length=hop_length)

### midi visualization ###
from Midi import midi_matrix
midi_mat = midi_matrix(midi_filename, min_pitch=pitch_min, max_pitch=pitch_max)

### NMF ###
V = np.abs(X)

## custom initialisation ##
W_zero = np.zeros((V.shape[0], n_components)).transpose()
threshold = 0.4
index = 0
#pitch = pitch_min
for comp in W_zero:
    h = 1
    p = pitches[index]
    while midi_to_hz(p) < W_zero.shape[1]:
        for freq in range(int(midi_to_hz(p-threshold)*n_fft/sr), int(midi_to_hz(p+threshold)*n_fft/sr)):
            if freq < W_zero.shape[1]:
                comp[freq] = 1.0 / h
        p += 12
        h += 1
    index += 1

W_zero = W_zero.transpose()
H_zero = np.random.rand(n_components, V.shape[1])

from NMF import factorize
comps, acts = factorize(V, W_zero, H_zero)

# filtering activations
filter_threshold = np.max(acts) / 5
acts[acts < filter_threshold] = 0

# visualisation matters
import matplotlib.pyplot as plt
from librosa.display import specshow
import matplotlib.gridspec as gridspec

plt.close('all')

#plt.subplot2grid((4, 2), (0,0))
#specshow(W_zero, sr=sr, hop_length=n_fft/4, n_yticks=10, y_axis='linear')
#plt.title('Initialised components')

#plt.subplot2grid((4, 2), (0,1))
#specshow(H_zero, sr=sr, x_axis='time')
#plt.title('Randomly initialised activations')

#plt.subplot2grid((3, 2), (0,0), colspan=2)
#specshow(V, sr=sr, x_axis='time', y_axis='linear')
#plt.colorbar()
#plt.title('Input power spectrogram')

plt.subplot2grid((1, 2), (0,0))
specshow(midi_mat, n_yticks=25, y_axis='cqt_note', fmin=midi_to_hz(pitch_min))
plt.xlabel('Ticks')
plt.title('Midi reference')

plt.subplot2grid((1, 2), (0,1))
specshow(acts, sr=sr, hop_length=hop_length, n_yticks=25, y_axis='cqt_note', x_axis='time', fmin=midi_to_hz(pitch_min))
plt.colorbar()
plt.ylabel('Components')
plt.title('Determined Activations')

#plt.subplot2grid((3, 2), (2,0), colspan=2)
#V_approx = comps.dot(acts)
#specshow(V_approx, sr=sr, x_axis='time', y_axis='linear')
#plt.colorbar()
#plt.title('Reconstructed spectrum')

plt.tight_layout()

plt.show()

