#!/usr/bin/python

from sys import argv

filename = argv[1]
midi_filename = "/home/martin/uni/music/miniproject/reference/ty_januarMINp_align.mid"
if len(argv) > 2:
    midi_filename = argv[2]

n_components=None
if len(argv) > 3:
    n_components = int(argv[3])

from librosa import load, cqt, logamplitude, note_to_midi, note_to_hz
import numpy as np

# load an audio file (with samplerate)
x, sr = load(filename)

# compute constant-Q transform (~ pitch-based STFT)
#hop_size = 512
pitch_max = note_to_midi('D5')
pitch_min = 'B3'
pitch_min_number = note_to_midi(pitch_min)
C = cqt(x, sr=sr, fmin=note_to_hz(pitch_min), n_bins=pitch_max-pitch_min_number)

# try some midi visualization
from Midi import midi_matrix

midi_mat = midi_matrix(midi_filename, min_pitch=note_to_midi(pitch_min))

# NMF

#V = np.log10(1 + 100000 * C**2)
V = np.abs(C).transpose()

W_zero = np.zeros((pitch_max - pitch_min_number, pitch_max - pitch_min_number))
pitch = pitch_min_number
for comp in W_zero:
    comp[pitch-pitch_min_number] = 1.0
    p = pitch + 12
    while p < W_zero.shape[1] - 2:
        for epsilon in range(-2, 2):
            comp[p - pitch_min + epsilon] = 1.0
        p += 12
        
H_zero = np.random.rand(V.shape[0], pitch_max - pitch_min_number)

print V.shape

from sklearn.decomposition import NMF

model = NMF(init='custom', n_components=pitch_max-pitch_min_number)
comps = model.fit_transform(V, W=H_zero, H=W_zero)
acts = model.components_

#from librosa.decompose import decompose

#comps, acts = decompose(V, n_components=n_components, sort=True)

# visualisation matters
import matplotlib.pyplot as plt
from librosa.display import specshow
import matplotlib.gridspec as gridspec

plt.close('all')

plt.subplot2grid((4, 2), (0,0), colspan=2)
specshow(midi_mat, sr=sr, x_axis='time', y_axis='cqt_note')
plt.title('midi visualization')

plt.subplot2grid((4, 2), (1,0), colspan=2)
specshow(V.transpose(), sr=sr, x_axis='time', y_axis='cqt_note')
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
V_approx = comps.dot(acts).transpose()
specshow(V_approx, sr=sr, x_axis='time', y_axis='cqt_note')
#plt.ylabel('pitches')
plt.colorbar(format='%+2.0f dB')
plt.title('Reconstructed spectrum')

plt.tight_layout()

plt.show()

