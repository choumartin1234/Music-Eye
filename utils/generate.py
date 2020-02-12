import numpy as np
import music21 as m21

m21.environment.set('musicxmlPath', '/usr/bin/musescore')
m21.environment.set('midiPath', '/usr/local/bin/timidity')
# soundfont /usr/share/sounds/sf2/FluidR3_GM.sf2


def generate_wave(freqs, fs, length, amplitudes=None, phases=None):
    # type: (list, float, int, list, list) -> ndarray
    r"""
    Returns overlapped sine waves whose frequencies are `freqs`.

    Args:
        freqs (list): a list of frequencies of target synthetic wave
        fs (float): sample frequency
        length (int): the length of the return ndarray
        amplitudes (list): the amplitudes for each frequency. Default: None (the
            amplitudes will be ones)
        phases (list): the initial phases for each frequency. Default: None (the
            phases will be ones)
    
    Example::

        >>> generate_wave([1], 2., 3)
        array([0.        , 0.47942554, 0.84147098]) 
    """
    freqs = np.asarray(freqs).reshape(-1, 1)
    if amplitudes is None:
        amplitudes = np.ones_like(freqs)
    if phases is None:
        phases = np.zeros_like(freqs)
    x = np.arange(length) / fs
    return np.sum(amplitudes * np.sin(2 * np.pi * freqs * x + phases), axis=0)


def generate_note(pitch='D#4', duration='half'):
    r"""
    Generate a musci21 note.
    """
    n = note.Note(pitch)
    n.duration.type = duration
    # n.show('midi')
    return n
