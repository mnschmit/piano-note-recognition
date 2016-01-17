#!/usr/bin/python

usage='''
Usage: show_own_NMF.py filename.wav [pitch_min pitch_max filtering]

       Mandatory argument : file to factorize
       Optional arguments : pitch_min (smallest pitch considered), pitch_max (biggest pitch considered), filtering (true or false) 
'''

import sys

if len(sys.argv) <= 1:
    print usage
    sys.exit(-1)

from librosa import load, stft, logamplitude, note_to_midi, midi_to_hz
import numpy as np

filename = sys.argv[1]

pitch_min = note_to_midi('C1')
if len(sys.argv) > 2:
    pitch_min = note_to_midi(sys.argv[2])

pitch_max = note_to_midi('C7')
if len(sys.argv) > 3:
    pitch_max = note_to_midi(sys.argv[3])

pitches = range(pitch_min, pitch_max + 1)

filtering = True
if len(sys.argv) > 4:
    if sys.argv[4] == "false":
        filtering = False 
    elif sys.argv[4] == "true":
        filtering = True
    else: 
        print "Error reading filtering argument. Assuming true."

### main program ###
x, sr = load(filename)

# compute normal STFT
n_components = len(pitches)
n_fft = 2048
hop_length = n_fft # big hop_length
X = stft(x, n_fft=n_fft, hop_length=hop_length)


### NMF ###
V = np.abs(X)

## custom initialisation ##
W_zero = np.zeros((V.shape[0], n_components)).transpose()
threshold = 0.4
index = 0
for comp in W_zero:
    h = 1
    p = pitches[index]
    while int(midi_to_hz(p)*n_fft/sr) < W_zero.shape[1]:
        for freq in range(int(midi_to_hz(p-threshold)*n_fft/sr), int(midi_to_hz(p+threshold)*n_fft/sr)):
            if freq < W_zero.shape[1]:
                comp[freq] = 1.0 / h
        p += 12
        h += 1
    index += 1

W_zero = W_zero.transpose()
H_zero = np.ones((n_components, V.shape[1]))

from NMF import factorize
comps, acts = factorize(V, W_zero, H_zero)

# filtering activations
if filtering:
    filter_threshold = np.max(acts) / 5

    for i in range(1, acts.shape[0]):
        for j in range(0, acts.shape[1]):
            if acts[i-1][j] > filter_threshold and acts[i-1][j] > acts[i][j]:
                acts[i-1][j] += acts[i][j]
                acts[i][j] = 0

    acts[acts < filter_threshold] = 0

# visualisation matters
import matplotlib.pyplot as plt
from librosa.display import specshow
import matplotlib.gridspec as gridspec

plt.close('all')

plt.subplot2grid((2, 2), (0, 0), colspan=2)
specshow(V, sr=sr, hop_length=hop_length, n_yticks=25, x_axis='time', y_axis='linear')
plt.colorbar()
plt.title('Input power spectrogram')

#plt.subplot2grid((2, 2), (0,1))
#specshow(W_zero, sr=sr, hop_length=hop_length, n_yticks=25, n_xticks=25, x_axis='frames', y_axis='linear')
##plt.colorbar()
#plt.xlabel('Components')
#plt.title('Initialised Components')

plt.subplot2grid((2, 2), (1,0))
specshow(comps, sr=sr, hop_length=hop_length, n_yticks=25, n_xticks=25, x_axis='frames', y_axis='linear')
#plt.colorbar()
plt.xlabel('Components')
plt.title('Learned Components')

plt.subplot2grid((2, 2), (1,1))
specshow(acts, sr=sr, hop_length=hop_length, n_yticks=25, y_axis='cqt_note', x_axis='time', fmin=midi_to_hz(pitch_min))
plt.colorbar()
plt.ylabel('Components')
plt.title('Determined Activations')

plt.tight_layout()

plt.show()

