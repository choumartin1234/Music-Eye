import matplotlib.pyplot as plt
import numpy as np
import librosa
import resampy
#from .torchvggish.torchvggish import vggish_input

def spectrum(data, fs=44100, target_fs=12800, hop_length=128, cqt=True, fmin=27.5, n_bins=87*4+1, bins_per_octave=4*12):
    if fs != target_fs:
        data = resampy.resample(data, fs, target_fs)
    if cqt:
        return np.log2(cqt_spectrum(data, target_fs, hop_length, fmin, n_bins, bins_per_octave) + 0.01).T
    else:
        return vggish_input.waveform_to_examples(data, target_fs, return_tensor=False)[0]

def cqt_spectrum(data, fs, hop_length, fmin, n_bins, bins_per_octave):
    return np.abs(librosa.cqt(data, fs, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave))
